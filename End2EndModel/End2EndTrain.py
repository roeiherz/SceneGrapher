import sys
from random import shuffle
import math
from keras import backend as K
import Scripts
from End2EndModel import End2EndModel
from FeaturesExtraction.Lib.Config import Config
from FeaturesExtraction.Utils.Boxes import BOX, find_union_box
from FeaturesExtraction.Utils.data import process_to_detections
from DesignPatterns.Detections import Detections

sys.path.append("..")
import itertools
import csv
from FeaturesExtraction.Utils.Utils import get_time_and_date, get_img_resize, get_img, get_mask_from_object
from LanguageModule import LanguageModule
from Utils.Utils import create_folder
import cPickle
from multiprocessing import Process
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
import os
import inspect

__author__ = 'roeih'

# feature sizes
VISUAL_FEATURES_PREDICATE_SIZE = 2048
VISUAL_FEATURES_OBJECT_SIZE = 2048
EMBED_SIZE = 10
NOF_PREDICATES = 51
NOF_OBJECTS = 150
# apply gradients every batch size
BATCH_SIZE = 100
# save model every number of iterations
SAVE_MODEL_ITERATIONS = 10
# test every number of iterations
TEST_ITERATIONS = 1
# Graph csv logger
CSVLOGGER = "training.log"
# negative vs positive factor
POS_NEG_FACTOR = 3.3
POS_NEG_RATIO = 1
# percentage of test data
TEST_PERCENT = 10


def get_csv_logger(tf_graphs_path, timestamp):
    """
    This function writes csv logger
    :param timestamp: time stamp directory
    :param tf_graphs_path: path directory for tf_graphs
    :return:
    """
    tf_graphs_path = os.path.join(tf_graphs_path, timestamp)
    create_folder(tf_graphs_path)
    Logger().log("Folder {0} has been created for CSV logger".format(tf_graphs_path))
    csv_file = open(os.path.join(tf_graphs_path, CSVLOGGER), "wb")
    csv_writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
    csv_writer.writeheader()
    return csv_writer, csv_file


def pre_process_data(entity, hierarchy_mapping, config):
    """
    This function is a generator for Predicate with Detections with batch-size
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    """
    imgs = []
    labels = []
    url = entity.image.url

    # Create object pairs
    objects_pairs = list(itertools.product(entity.objects, repeat=2))

    # Create a dict with key as pairs - (subject, object) and their values are predicates use for labels
    relations_dict = {}

    # Create a dict with key as pairs - (subject, object) and their values are relation index_id
    relations_filtered_id_dict = {}
    for relation in entity.relationships:
        relations_dict[(relation.subject.id, relation.object.id)] = relation.predicate
        relations_filtered_id_dict[(relation.subject.id, relation.object.id)] = relation.filtered_id

    if len(relations_dict) != len(entity.relationships):
        Logger().log("**Error in entity image {0} with number of {1} relationship and {2} of relations_dict**"
                     .format(entity.image.id, len(entity.relationships), len(relations_dict)))

    # Get image
    img = get_img(url, download=True)

    if img is None:
        Logger().log("Coulden't get the image in url {}".format(url))
        return None, None

    for relation in objects_pairs:

        subject = relation[0]
        object = relation[1]

        # Get the label of object
        if (subject.id, object.id) in relations_dict:
            label = relations_dict[(subject.id, object.id)]

        else:
            # Negative label
            label = "neg"

        # Check if it is a correct label
        if label not in hierarchy_mapping.keys():
            Logger().log("WARNING: label isn't familiar")
            return None

        # Get the label uuid
        label_id = hierarchy_mapping[label]

        # Create the y labels as a one hot vector
        y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

        # Get Subject and Object boxes
        # Subject
        # Get the mask: a dict with {x1,x2,y1,y2}
        subject_mask = get_mask_from_object(subject)
        # Saves as a box
        subject_box = np.array([subject_mask['x1'], subject_mask['y1'], subject_mask['x2'], subject_mask['y2']])

        # Object
        # Get the mask: a dict with {x1,x2,y1,y2}
        object_mask = get_mask_from_object(object)
        # Saves as a box
        object_box = np.array([object_mask['x1'], object_mask['y1'], object_mask['x2'], object_mask['y2']])

        # Fill HeatMap
        heat_map_subject = np.zeros(img.shape)
        heat_map_subject[subject_box[BOX.Y1]: subject_box[BOX.Y2], subject_box[BOX.X1]: subject_box[BOX.X2],
        :] = 255
        heat_map_object = np.zeros(img.shape)
        heat_map_object[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2],
        :] = 255

        # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
        box = find_union_box(subject_box, object_box)

        # Cropping the patch from the image.
        patch_predicate = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
        patch_heatmap_heat_map_subject = heat_map_subject[box[BOX.Y1]: box[BOX.Y2],
                                         box[BOX.X1]: box[BOX.X2], :]
        patch_heatmap_heat_map_object = heat_map_object[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2],
                                        :]

        # Resize the image according the padding method
        resized_img = get_img_resize(patch_predicate, config.crop_width, config.crop_height,
                                     type=config.padding_method)
        resized_heatmap_subject = get_img_resize(patch_heatmap_heat_map_subject, config.crop_width,
                                                 config.crop_height, type=config.padding_method)
        resized_heatmap_object = get_img_resize(patch_heatmap_heat_map_object, config.crop_width,
                                                config.crop_height, type=config.padding_method)

        # Concatenate the heat-map to the image in the kernel axis
        resized_img = np.concatenate((resized_img, resized_heatmap_subject[:, :, :1]), axis=2)
        resized_img = np.concatenate((resized_img, resized_heatmap_object[:, :, :1]), axis=2)

        # Expand dimensions - add batch dimension for the numpy
        resized_img = np.expand_dims(resized_img, axis=0)
        y_labels = np.expand_dims(y_labels, axis=0)

        imgs.append(np.copy(resized_img))
        labels.append(np.copy(y_labels))

    return np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)


def test(labels_predicate, labels_object, out_belief_predicate_val, out_belief_object_val):
    """
    returns a dictionary with statistics about object, predicate and relationship accuracy in this image
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_object: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_belief_predicate_val: belief of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_belief_object_val: belief of image objects - shape (N, NOF_OBJECTS)
    :return: see description
    """
    predicats_gt = np.argmax(labels_predicate, axis=2)
    objects_gt = np.argmax(labels_object, axis=1)
    predicats_pred = np.argmax(out_belief_predicate_val, axis=2)
    predicats_pred_no_neg = np.argmax(out_belief_predicate_val[:, :, :NOF_PREDICATES - 1], axis=2)
    objects_pred = np.argmax(out_belief_object_val, axis=1)

    results = {}
    # number of objects
    results["obj_total"] = objects_gt.shape[0]
    # number of predicates / relationships
    results["predicates_total"] = predicats_gt.shape[0] * predicats_gt.shape[1]
    # number of positive predicates / relationships
    pos_indices = np.where(predicats_gt != NOF_PREDICATES - 1)
    results["predicates_pos_total"] = pos_indices[0].shape[0]

    # number of object correct predictions
    results["obj_correct"] = np.sum(objects_gt == objects_pred)
    # number of correct predicate
    results["predicates_correct"] = np.sum(predicats_gt == predicats_pred)
    # number of correct positive predicates
    predicats_gt_pos = predicats_gt[pos_indices]
    predicats_pred_pos = predicats_pred_no_neg[pos_indices]
    results["predicates_pos_correct"] = np.sum(predicats_gt_pos == predicats_pred_pos)
    # number of correct relationships
    object_true_indices = np.where(objects_gt == objects_pred)
    predicats_gt_true = predicats_gt[object_true_indices[0], :][:, object_true_indices[0]]
    predicats_pred_true = predicats_pred[object_true_indices[0], :][:, object_true_indices[0]]
    predicats_pred_true_pos = predicats_pred_no_neg[object_true_indices[0], :][:, object_true_indices[0]]
    results["relationships_correct"] = np.sum(predicats_gt_true == predicats_pred_true)
    # number of correct positive relationships
    pos_true_indices = np.where(predicats_gt_true != NOF_PREDICATES - 1)
    predicats_gt_pos_true = predicats_gt_true[pos_true_indices]
    predicats_pred_pos_true = predicats_pred_true_pos[pos_true_indices]
    results["relationships_pos_correct"] = np.sum(predicats_gt_pos_true == predicats_pred_pos_true)

    return results


def predicate_class_recall(labels_predicate, out_belief_predicate_val, k=5):
    """
    Predicate Classification - Examine the model performance on predicates classification in isolation from other factors
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param out_belief_predicate_val: belief of image predicates - shape (N, N, NOF_PREDICATES)
    :param k: k most confident predictions to consider
    :return: correct vector (number of times predicate gt appears in top k most confident predicates),
             total vector ( number of gts per predicate)
    """
    correct = np.zeros(NOF_PREDICATES)
    total = np.zeros(NOF_PREDICATES)

    # one hot vector to actual gt labels
    predicates_gt = np.argmax(labels_predicate, axis=2)

    # number of objects in the image
    N = out_belief_predicate_val.shape[0]

    # run over each prediction
    for subject_index in range(N):
        for object_index in range(N):
            # get predicate class
            predicate_class = predicates_gt[subject_index][object_index]
            # get predicate probabilities
            predicate_prob = out_belief_predicate_val[subject_index][object_index]

            max_k_predictions = np.argsort(predicate_prob)[-k:]
            found = np.where(predicate_class == max_k_predictions)[0]
            if len(found) != 0:
                correct[predicate_class] += 1
            total[predicate_class] += 1

    return correct, total


def train(name="test",
          nof_iterations=100,
          learning_rate=0.1,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          rnn_steps=1,
          loss_func="all",
          lr_object_coeff=1,
          including_object=False,
          include_bb=False,
          layers=[],
          reg_factor=0.03,
          gpu=0):
    """

    :param files_test_list: the list of test files
    :param files_train_list: the list of train files
    :param name: the name of the module
    :param nof_iterations: num of iterations
    :param learning_rate: the lr
    :param learning_rate_steps: the num of steps which the lr will be updated
    :param learning_rate_decay: the decay of the lr
    :param load_module_name: the module file name
    :param use_saved_module: to load or start from scratch module
    :param timesteps: how many RNNs
    :param gpu: which GPU to use
    :return:
    """

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # print train params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    Logger().log('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        Logger().log("    %s = %s" % (i, values[i]))

    Logger().log("    %s = %s" % ("POS_NEG_RATIO", POS_NEG_RATIO))

    # Load class config
    config = Config(gpu)
    # Print to the logger the config params
    config.config_logger()

    # Create Module
    e2e_module = End2EndModel(config=config, nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                              visual_features_predicate_size=VISUAL_FEATURES_PREDICATE_SIZE,
                              visual_features_object_size=VISUAL_FEATURES_OBJECT_SIZE, is_train=True,
                              learning_rate=learning_rate, learning_rate_steps=learning_rate_steps,
                              learning_rate_decay=learning_rate_decay,
                              rnn_steps=rnn_steps,
                              loss_func=loss_func,
                              lr_object_coeff=lr_object_coeff,
                              including_object=including_object,
                              include_bb=include_bb,
                              layers=layers)

    # @todo: clean
    # # get input place holders
    # img_inputs_ph = e2e_module.get_inputs_placeholders()
    # # get labels place holders
    # labels_ph = e2e_module.get_labels_placeholders()
    # # get coeff place holders
    # # coeff_loss_ph = e2e_module.get_coeff_placeholders()
    # # get learning rate place holder
    # lr_ph = e2e_module.get_lr_placeholder()
    # # get loss and train step
    # loss, gradients, grad_placeholder, train_step = e2e_module.module_loss()
    # # get module output
    # # accuracy = e2e_module.get_output()
    # # get logits (likelihood)
    # logits = e2e_module.get_logits()

    # get input place holders
    img_inputs_ph, belief_object_ph, extended_belief_object_shape_ph, \
    visual_features_predicate_ph, visual_features_object_ph, num_objects_ph = e2e_module.get_in_ph()
    # get labels place holders
    labels_predicate_ph, labels_object_ph, labels_coeff_loss_ph = e2e_module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = e2e_module.get_module_loss()
    # get module output
    out_predicate_probes, out_object_probes = e2e_module.get_output()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    variables = tf.contrib.slim.get_variables_to_restore()
    # [v for v in variables if "global_obj_pred_attention" not in v.name and "transpose" not in v.name and "Add_3_h" not in v.name]
    variables_to_restore = variables
    saver = tf.train.Saver(variables_to_restore, max_to_keep=None)
    # Get timestamp
    timestamp = get_time_and_date()
    # # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(max_to_keep=None)

    # Define Summaries
    tf_logs_path = FilesManager().get_file_path("e2e_module.train.tf_logs")
    summary_writer = tf.summary.FileWriter(tf_logs_path, graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()
    tf_graphs_path = FilesManager().get_file_path("e2e_module.train.tf_graphs")
    csv_writer, csv_file = get_csv_logger(tf_graphs_path, timestamp)

    Logger().log("Start Training")

    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = FilesManager().get_file_path("e2e_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        # saver.save(sess, module_path_load + '/model.ckpt', 0)

        if os.path.exists(module_path_load + ".index") and use_saved_module:
            saver.restore(sess, module_path_load)
            Logger().log("Model restored.")
        else:
            create_folder(os.path.join(module_path, timestamp))
            # K.set_session(sess)
            model = tf.contrib.keras.models.Model(inputs=e2e_module.img_inputs_ph, outputs=e2e_module.output_resnet50,
                                                  name='resnet50')
            net_weights = "/home/roeih/SceneGrapher/FilesManager/FeaturesExtraction/PredicatesMaskCNN/Fri_Oct_27_22:41:05_2017/model_vg_resnet50.hdf5"
            model.load_weights(net_weights, by_name=True)
            # sess = tf.contrib.keras.backend.get_session()
            saver = tf.train.Saver()
            module_path_load = os.path.join(module_path, timestamp)
            saver.save(sess, module_path_load + '/resnet50_model.ckpt', 0)
            # sess.run(init)

        # Get the entities
        entities_path = FilesManager().get_file_path("data.visual_genome.detections_v4")
        files_train_list = ["Sat_Nov_11_21:59:10_2017"]
        files_test_list = ["Sat_Nov_11_21:59:10_2017"]

        if files_train_list is None or len(files_train_list) == 0:
            Logger().log("Error: No training data")
            return None

        if files_test_list is None or len(files_test_list) == 0:
            Logger().log("Error: No testing data")
            return None

        # @todo: clean
        # get mini data to filter
        # img_ids = Scripts.get_img_ids()

        # Load hierarchy_mappings
        # @todo: clean
        # hierarchy_mapping_objects = FilesManager().load_file("data.visual_genome.hierarchy_mapping_objects")
        hierarchy_mapping_predicates = FilesManager().load_file("data.visual_genome.hierarchy_mapping_predicates")

        # @todo: clean
        # Process relations to numpy Detections dtype
        # vfunc = np.vectorize(lambda url: int(url.split("/")[-1].split('.')[0]))
        # detections_train = process_to_detections(None, detections_file_name="full_detections_test")
        # train_img_ids = set(vfunc(detections_train[Detections.Url]))
        # detections_test = process_to_detections(None, detections_file_name="full_detections_test")
        # test_img_ids = set(vfunc(detections_test[Detections.Url]))

        # Object embedding
        embed_obj = FilesManager().load_file("language_module.word2vec.object_embeddings")
        embed_pred = FilesManager().load_file("language_module.word2vec.predicate_embeddings")
        # Concat negative represntation
        embed_pred = np.concatenate((embed_pred, np.zeros(embed_pred[:1].shape)), axis=0)

        # Create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1

        # module
        lr = learning_rate
        best_test_loss = -1
        for epoch in range(1, nof_iterations):
            try:
                accum_results = None
                total_loss = 0
                steps = []
                # Shuffle entities groups
                # shuffle(files_train_list)
                # read data
                file_index = -1
                for file_dir in files_train_list:
                    files = os.listdir(os.path.join(entities_path, file_dir))
                    for file_name in files:

                        # Load only entities
                        # if ".log" in file_name or "lang" in file_name:
                        #     continue
                        if "language_language_language" not in file_name:
                            continue

                        file_path = os.path.join(entities_path, file_dir, file_name)
                        file_handle = open(file_path, "rb")
                        train_entities = cPickle.load(file_handle)
                        file_handle.close()
                        # shuffle(train_entities)
                        for entity in train_entities:
                            try:

                                # @todo: clean
                                # if entity.image.id != 2416509:
                                #     continue

                                if len(entity.relationships) == 0:
                                    continue

                                indices = np.arange(entity.predicates_probes.shape[0])

                                # Set diagonal to be neg
                                set_diagonal_neg(entity, predicate_neg, indices)

                                # Get shape of extended object to be used by the module
                                extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                                extended_belief_object_shape[2] = NOF_OBJECTS
                                extended_obj_bb_shape = np.asarray(entity.predicates_probes.shape)
                                extended_obj_bb_shape[2] = 4

                                # Get objects bounding boxes
                                obj_bb = get_objects_bb(entity.objects)

                                # filter non mixed cases
                                predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                                if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                                        predicates_neg_labels) == 0:
                                    continue

                                # PredCls or SGCls task
                                if including_object:
                                    in_object_belief = entity.objects_outputs_with_no_activations
                                else:
                                    in_object_belief = entity.objects_labels * 1000

                                # Give lower weights to negatives
                                coeff_factor = get_coeff_mat(entity, predicates_neg_labels, indices)

                                predicate_inputs, predicate_outputs = pre_process_data(entity,
                                                                                       hierarchy_mapping_predicates,
                                                                                       config)

                                if predicate_inputs is None or predicate_outputs is None:
                                    logger.log(
                                        "Error: No predicate inputs or predicate labels in {}".format(entity.image.url))
                                    continue

                                # Create the feed dictionary
                                feed_dict = {img_inputs_ph: predicate_inputs, belief_object_ph: in_object_belief,
                                             extended_belief_object_shape_ph: extended_belief_object_shape,
                                             e2e_module.extended_obj_bb_shape_ph: extended_obj_bb_shape,
                                             e2e_module.obj_bb_ph: obj_bb,
                                             e2e_module.word_embed_objects: embed_obj,
                                             e2e_module.word_embed_predicates: embed_pred,
                                             e2e_module.phase_ph: True,
                                             labels_predicate_ph: entity.predicates_labels,
                                             labels_object_ph: entity.objects_labels,
                                             labels_coeff_loss_ph: coeff_factor.reshape((-1)), e2e_module.lr_ph: lr,
                                             num_objects_ph: len(entity.objects)
                                             }

                                # Run the network
                                out_predicate_probes_val, out_object_probes_val, loss_val, gradients_val = sess.run(
                                    [out_predicate_probes, out_object_probes, loss, gradients],
                                    feed_dict=feed_dict)

                                if math.isnan(loss_val):
                                    logger.log(
                                        "Error: loss is NAN in epoch {0} in image {1}".format(epoch, entity.image.url))
                                    continue

                                # set diagonal to be neg (in order not to take into account in statistics)
                                out_predicate_probes_val[indices, indices, :] = predicate_neg

                                # Append gradient to list (will be applied as a batch of entities)
                                steps.append(gradients_val)
                                # Calculates loss
                                total_loss += loss_val

                                results = test(entity.predicates_labels, entity.objects_labels,
                                               out_predicate_probes_val,
                                               out_object_probes_val)

                                # Accumulate results
                                if accum_results is None:
                                    accum_results = results
                                else:
                                    for key in results:
                                        accum_results[key] += results[key]

                                # Update gradients in each epoch
                                if len(steps) == BATCH_SIZE:
                                    # Apply steps
                                    step = steps[0]
                                    feed_grad_apply_dict = {grad_placeholder[j][0]: step[j][0] for j in
                                                            xrange(len(grad_placeholder))}
                                    for i in xrange(1, len(steps)):
                                        step = steps[i]
                                        for j in xrange(len(grad_placeholder)):
                                            feed_grad_apply_dict[grad_placeholder[j][0]] += step[j][0]

                                    feed_grad_apply_dict[module.lr_ph] = lr
                                    sess.run([train_step], feed_dict=feed_grad_apply_dict)
                                    steps = []

                            except Exception as e:
                                logger.log(
                                    "Error: problem in Train. Epoch: {0}, image id: {1} Exception: {2}".format(epoch,
                                                                                                               entity.image.id,
                                                                                                               str(
                                                                                                                   e)))
                                continue

                # endregion
                # Finished training - one Epoch
                if epoch == 1:
                    obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
                    predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results[
                        'predicates_pos_total']
                    relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                        'predicates_pos_total']
                    logger.log("iter %d.%d - obj %f - pred %f - relation %f" %
                               (epoch, file_index, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy))

                # Print stat
                obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
                predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results[
                    'predicates_pos_total']
                predicate_all_accuracy = float(accum_results['predicates_correct']) / accum_results['predicates_total']
                relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                    'predicates_pos_total']
                relationships_all_accuracy = float(accum_results['relationships_correct']) / accum_results[
                    'predicates_total']

                logger.log("iter %d - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - lr %f" %
                           (epoch, total_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy, lr))

                # region Testing
                if epoch % TEST_ITERATIONS == 0:
                    total_test_loss = 0
                    accum_test_results = None
                    correct_predicate = 0
                    total_predicate = 0
                    for file_dir in files_test_list:
                        files = os.listdir(os.path.join(entities_path, file_dir))
                        for file_name in files:

                            # Load only entities
                            if ".log" in file_name:
                                continue

                            file_path = os.path.join(entities_path, file_dir, file_name)
                            file_handle = open(file_path, "rb")
                            test_entities = cPickle.load(file_handle)
                            file_handle.close()
                            for entity in test_entities:
                                try:

                                    if len(entity.relationships) == 0:
                                        continue

                                    indices = np.arange(entity.predicates_probes.shape[0])

                                    # Set diagonal to be neg
                                    set_diagonal_neg(entity, predicate_neg, indices)

                                    # Get shape of extended object to be used by the module
                                    extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                                    extended_belief_object_shape[2] = NOF_OBJECTS
                                    extended_obj_bb_shape = np.asarray(entity.predicates_probes.shape)
                                    extended_obj_bb_shape[2] = 4

                                    # Get objects bounding boxes
                                    obj_bb = get_objects_bb(entity.objects)

                                    # filter non mixed cases
                                    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
                                    if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                                            predicates_neg_labels) == 0:
                                        continue

                                    # PredCls or SGCls task
                                    if including_object:
                                        in_object_belief = entity.objects_outputs_with_no_activations
                                    else:
                                        in_object_belief = entity.objects_labels * 1000

                                    # Give lower weights to negatives
                                    coeff_factor = get_coeff_mat(entity, predicates_neg_labels, indices)

                                    predicate_inputs, predicate_outputs = pre_process_data(entity,
                                                                                           hierarchy_mapping_predicates,
                                                                                           config)

                                    if predicate_inputs is None or predicate_outputs is None:
                                        logger.log(
                                            "Error: No predicate inputs or predicate labels in {}".format(
                                                entity.image.url))
                                        continue

                                    # Create the feed dictionary
                                    feed_dict = {img_inputs_ph: predicate_inputs, belief_object_ph: in_object_belief,
                                                 extended_belief_object_shape_ph: extended_belief_object_shape,
                                                 e2e_module.extended_obj_bb_shape_ph: extended_obj_bb_shape,
                                                 e2e_module.obj_bb_ph: obj_bb,
                                                 e2e_module.word_embed_objects: embed_obj,
                                                 e2e_module.word_embed_predicates: embed_pred,
                                                 e2e_module.phase_ph: True,
                                                 labels_predicate_ph: entity.predicates_labels,
                                                 labels_object_ph: entity.objects_labels,
                                                 labels_coeff_loss_ph: coeff_factor.reshape((-1)), e2e_module.lr_ph: lr,
                                                 num_objects_ph: len(entity.objects)
                                                 }

                                    # Run the network
                                    out_predicate_probes_val, out_object_probes_val, loss_val = sess.run(
                                        [out_predicate_probes, out_object_probes, loss], feed_dict=feed_dict)

                                    # set diagonal to be neg (in order not to take into account in statistics)
                                    out_predicate_probes_val[indices, indices, :] = predicate_neg

                                    # Statistic
                                    total_test_loss += loss_val

                                    # statistics
                                    soft_max1 = np.exp(entity.predicates_outputs_beliefs_language1) / np.sum(
                                        np.exp(entity.predicates_outputs_beliefs_language1), axis=2, keepdims=True)
                                    out_predicate_probes_val = soft_max1  # entity.predicates_probes
                                    out_object_probes_val = entity.objects_probs
                                    results = test(entity.predicates_labels, entity.objects_labels,
                                                   out_predicate_probes_val, out_object_probes_val)

                                    # accumulate results
                                    if accum_test_results is None:
                                        accum_test_results = results
                                    else:
                                        for key in results:
                                            accum_test_results[key] += results[key]

                                    # Eval per predicate
                                    correct_predicate_image, total_predicate_image = predicate_class_recall(entity.predicates_labels,
                                                                                                            out_predicate_probes_val)
                                    correct_predicate += np.sum(correct_predicate_image[:NOF_PREDICATES-2])
                                    total_predicate += np.sum(total_predicate_image[:NOF_PREDICATES-2])

                                except Exception as e:
                                    logger.log(
                                        "Error: problem in Test. Epoch: {0}, image id: {1} Exception: {2}".format(epoch,
                                                                                                                  entity.image.id,
                                                                                                                  str(
                                                                                                                      e)))
                                    continue

                    # Print stats
                    obj_accuracy = float(accum_test_results['obj_correct']) / accum_test_results['obj_total']
                    predicate_pos_accuracy = float(accum_test_results['predicates_pos_correct']) / accum_test_results[
                        'predicates_pos_total']
                    predicate_all_accuracy = float(accum_test_results['predicates_correct']) / accum_test_results[
                        'predicates_total']
                    relationships_pos_accuracy = float(accum_test_results['relationships_pos_correct']) / \
                                                 accum_test_results[
                                                     'predicates_pos_total']
                    relationships_all_accuracy = float(accum_test_results['relationships_correct']) / \
                                                 accum_test_results[
                                                     'predicates_total']

                    logger.log(
                        "TEST Epoch - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - top5 %f" %
                        (total_test_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                         predicate_all_accuracy, relationships_all_accuracy,
                         float(correct_predicate) / total_predicate))

                    # Write to CSV logger
                    csv_writer.writerow({'epoch': epoch, 'loss': total_test_loss, 'object_acc': obj_accuracy,
                                         'pred_pos_acc': predicate_pos_accuracy,
                                         'rela_pos_acc': relationships_pos_accuracy,
                                         'all_pred_acc': predicate_all_accuracy,
                                         'all_rela_acc': relationships_all_accuracy,
                                         'top5': float(correct_predicate) / total_predicate})
                    csv_file.flush()

                    # save best module so far
                    if best_test_loss == -1 or total_test_loss < best_test_loss:
                        # Save the best module till 5 epoch as different name
                        if epoch < 5:
                            module_path_save = os.path.join(module_path, timestamp, name + "_best5_module.ckpt")
                            save_path = saver.save(sess, module_path_save)
                            logger.log("Model Best till 5 epoch saved in file: %s" % save_path)

                        module_path_save = os.path.join(module_path, timestamp, name + "_best_module.ckpt")
                        save_path = saver.save(sess, module_path_save)
                        logger.log("Model Best saved in file: %s" % save_path)
                        best_test_loss = total_test_loss

                # endregion
                # Finished Testing

                # Save module
                if epoch % SAVE_MODEL_ITERATIONS == 0 or epoch < 6:
                    module_path_save = os.path.join(module_path, timestamp, name + "_{}_module.ckpt".format(epoch))
                    save_path = saver.save(sess, module_path_save)
                    logger.log("Model saved in file: %s" % save_path)

                # Update learning rate decay
                if epoch % learning_rate_steps == 0:
                    lr *= learning_rate_decay
            except Exception as e:
                logger.log("Error: problem in epoch: {0} with: {1}".format(epoch, str(e)))
                continue

        # Save module
        module_path_save = os.path.join(module_path, timestamp, name + "_end_module.ckpt")
        save_path = saver.save(sess, module_path_save)
        logger.log("Model saved in file: %s" % save_path)

        # Close csv logger
        csv_file.close()


def get_coeff_mat(entity, predicates_neg_labels, indices):
    """
    This function returns the coeff factor matrix
    :param entity:
    :param predicates_neg_labels:
    :param indices:
    :return:
    """
    coeff_factor = np.ones(predicates_neg_labels.shape)
    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
        predicates_neg_labels) / POS_NEG_FACTOR
    coeff_factor[predicates_neg_labels == 1] *= factor
    coeff_factor[indices, indices] = 0
    coeff_factor[predicates_neg_labels == 1] = 0
    return coeff_factor


def set_diagonal_neg(entity, predicate_neg, indices):
    """
    This function set diagonal to be negative
    :param entity:
    :param predicate_neg:
    :return:
    """

    entity.predicates_outputs_beliefs_language1[indices, indices, :] = predicate_neg
    entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
    entity.predicates_labels[indices, indices, :] = predicate_neg
    entity.predicates_probes[indices, indices, :] = predicate_neg


def get_objects_bb(objects):
    """
    This function creates list of objects bounding boxes
    :param objects:
    :return:
    """
    obj_bb = np.zeros((len(objects), 4))
    for obj_id in range(len(objects)):
        obj_bb[obj_id][0] = objects[obj_id].x / 1200.0
        obj_bb[obj_id][1] = objects[obj_id].y / 1200.0
        obj_bb[obj_id][2] = (objects[obj_id].x + objects[obj_id].width) / 1200.0
        obj_bb[obj_id][3] = (objects[obj_id].y + objects[obj_id].height) / 1200.0

        # Pre-processing entities to get RNN inputs and outputs
    return obj_bb


def get_coeff_factor(entity, indices):
    """
    This function returns coeff matrix for input to the BI-RNN
    :param entity: Entity class
    :param indices: a numpy array of indices
    :return: coeff_matrix [num_objects, num_objects, 51] - positives coeff: 1, negatives get coeff: ratio, diag coeff: 0
    """
    # Filter non mixed cases
    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]
    # Give lower weight to negatives
    coeff_factor = np.ones(predicates_neg_labels.shape)
    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / \
             np.sum(predicates_neg_labels) / POS_NEG_FACTOR
    coeff_factor[predicates_neg_labels == 1] *= factor
    coeff_factor[indices, indices] = 0
    return coeff_factor


def set_diag_to_negatives(entity, predicate_neg):
    """
    This function set diagonal to negative
    :param entity: Entity class
    :param predicate_neg: a numpy array of [0,0,....,1]
    :return:
    """
    indices = np.arange(entity.predicates_probes.shape[0])
    entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
    entity.predicates_labels[indices, indices, :] = predicate_neg
    entity.predicates_probes[indices, indices, :] = predicate_neg
    return indices


if __name__ == "__main__":
    filemanager = FilesManager()
    logger = Logger()

    params = filemanager.load_file("e2e_module.train.params")
    nof_processes = params["nof_p"]

    processes = []
    for process in range(1, nof_processes + 1):
        process_params = params[process]
        name = process_params["name"]
        learning_rate = process_params["learning_rate"]
        learning_rate_steps = process_params["learning_rate_steps"]
        learning_rate_decay = process_params["learning_rate_decay"]
        nof_iterations = process_params["nof_iterations"]
        load_model_name = process_params["load_model_name"]
        use_saved_model = process_params["use_saved_model"]
        rnn_steps = process_params["rnn_steps"]
        gpu = process_params["gpu"]
        files_train_list = process_params["files_train"]
        files_test_list = process_params["files_test"]
        lr_object_coeff = process_params["lr_object_coeff"]
        including_object = process_params["including_object"]
        include_bb = process_params["include_bb"]
        layers = process_params["layers"]
        reg_factor = process_params["reg_factor"]
        loss_func = process_params["loss_func"]

        train(name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
              use_saved_model, rnn_steps, loss_func, lr_object_coeff, including_object, include_bb, layers, reg_factor,
              gpu)

    # wait until all processes done
    for p in processes:
        p.join()
