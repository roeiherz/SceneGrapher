import sys

sys.path.append("..")

from random import shuffle
import math
from keras import backend as K
import Scripts
from End2EndModel import End2EndModel
from FeaturesExtraction.Lib.Config import Config
from FeaturesExtraction.Utils.Boxes import BOX, find_union_box
from FeaturesExtraction.Utils.data import process_to_detections
from DesignPatterns.Detections import Detections

import itertools
import csv
from FeaturesExtraction.Utils.Utils import get_time_and_date, get_img_resize, get_img, get_mask_from_object
from LanguageModule import LanguageModule
from Utils.Utils import create_folder
import cPickle
from multiprocessing import Process, Queue
import threading
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
import os
import inspect
from tensorflow.contrib import slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

NOF_PREDICATES = 51
NOF_OBJECTS = 150
# save model every number of iterations
SAVE_MODEL_ITERATIONS = 10
# test every number of iterations
TEST_ITERATIONS = 1
# Graph csv logger
CSVLOGGER = "training.log"


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


def test(labels_relation, labels_entity, out_confidence_relation_val, out_confidence_entity_val):
    """
    returns a dictionary with statistics about object, predicate and relationship accuracy in this image
    :param labels_relation: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_entity: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_confidence_relation_val: confidence of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_confidence_entity_val: confidence of image objects - shape (N, NOF_OBJECTS)
    :return: see description
    """
    relation_gt = np.argmax(labels_relation, axis=2)
    entity_gt = np.argmax(labels_entity, axis=1)
    relation_pred = np.argmax(out_confidence_relation_val, axis=2)
    relations_pred_no_neg = np.argmax(out_confidence_relation_val[:, :, :NOF_PREDICATES - 1], axis=2)
    entities_pred = np.argmax(out_confidence_entity_val, axis=1)

    # noinspection PyDictCreation
    results = {}
    # number of objects
    results["entity_total"] = entity_gt.shape[0]
    # number of predicates / relationships
    results["relations_total"] = relation_gt.shape[0] * relation_gt.shape[1]
    # number of positive predicates / relationships
    pos_indices = np.where(relation_gt != NOF_PREDICATES - 1)
    results["relations_pos_total"] = pos_indices[0].shape[0]

    # number of object correct predictions
    results["entity_correct"] = np.sum(entity_gt == entities_pred)
    # number of correct predicate
    results["relations_correct"] = np.sum(relation_gt == relation_pred)
    # number of correct positive predicates
    relations_gt_pos = relation_gt[pos_indices]
    relations_pred_pos = relations_pred_no_neg[pos_indices]
    results["relations_pos_correct"] = np.sum(relations_gt_pos == relations_pred_pos)
    # number of correct relationships
    entity_true_indices = np.where(entity_gt == entities_pred)
    relations_gt_true = relation_gt[entity_true_indices[0], :][:, entity_true_indices[0]]
    relations_pred_true = relation_pred[entity_true_indices[0], :][:, entity_true_indices[0]]
    relations_pred_true_pos = relations_pred_no_neg[entity_true_indices[0], :][:, entity_true_indices[0]]
    results["relationships_correct"] = np.sum(relations_gt_true == relations_pred_true)
    # number of correct positive relationships
    pos_true_indices = np.where(relations_gt_true != NOF_PREDICATES - 1)
    relations_gt_pos_true = relations_gt_true[pos_true_indices]
    relations_pred_pos_true = relations_pred_true_pos[pos_true_indices]
    results["relationships_pos_correct"] = np.sum(relations_gt_pos_true == relations_pred_pos_true)

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


class PreProcessWorker(threading.Thread):
    def __init__(self, module, train_images, relation_neg, queue, lr, pred_pos_neg_ratio, hierarchy_mapping_objects,
                 hierarchy_mapping_predicates, config, is_train):
        threading.Thread.__init__(self)
        self.train_images = train_images
        self.relation_neg = relation_neg
        self.queue = queue
        self.module = module
        self.lr = lr
        self.pred_pos_neg_ratio = pred_pos_neg_ratio
        self.hierarchy_mapping_objects = hierarchy_mapping_objects
        self.hierarchy_mapping_predicates = hierarchy_mapping_predicates
        self.config = config
        self.is_train = is_train
        self.size = len(train_images)

    def pre_process_entities_data(self, image, ind, img):
        """
        This function is a generator for Object with Detections with batch-size
        :param ind: index of an image of the whole images (used only for entities mixup Jitter)
        :param image: image entity VG
        :return: This function will return numpy array of training images, numpy array of labels
        """
        patches = []
        labels = []
        url = image.image.url
        indices = set(range(self.size))

        if img is None:
            Logger().log("Couldn't get the image in url {}".format(url))
            return None, None

        for entity in image.objects:

            # Get the lable of object
            label = entity.names[0]

            # Check if it is a correct label
            if label not in self.hierarchy_mapping_objects.keys():
                Logger().log("WARNING: label isn't familiar")
                return None

            # Get the label uuid
            label_id = self.hierarchy_mapping_objects[label]

            # Create the y labels as a one hot vector
            y_labels = np.eye(len(self.hierarchy_mapping_objects), dtype='uint8')[label_id]

            # Get the mask: a dict with {x1,x2,y1,y2}
            mask = get_mask_from_object(entity)

            # Cropping the patch from the image.
            patch = img[mask['y1']: mask['y2'], mask['x1']: mask['x2'], :]

            # Resize the image according the padding method
            resized_patch = get_img_resize(patch, self.config.crop_width, self.config.crop_height,
                                           type=self.config.padding_method)

            # Augment only in training
            if self.is_train and self.config.use_jitter:
                new_resized_patch = None

                # For mixup Jitter we need to create a new resize_img from another sample
                if False: #self.config.jitter.use_mixup:
                    all_indice_without_ind = list(indices - set([ind]))
                    # Pick different index from the data with no repetition
                    new_ind = np.random.choice(all_indice_without_ind)
                    new_object = self.train_images[new_ind]

                    new_img = get_img(new_object.url, download=True)
                    if new_img is None:
                        Logger().log("Coulden't get the image")
                        continue
                    # Get the mask: a dict with {x1,x2,y1,y2}
                    new_mask = get_mask_from_object(new_object)
                    # Cropping the patch from the image.
                    new_patch = new_img[new_mask['y1']: new_mask['y2'], new_mask['x1']: new_mask['x2'], :]
                    # Resize the image according the padding method
                    new_resized_patch = get_img_resize(new_patch, self.config.crop_width, self.config.crop_height,
                                                     type=self.config.padding_method)

                resized_patch = self.config.jitter.apply_jitter(resized_img=resized_patch, batchsize=self.size,
                                                                new_resized_img=new_resized_patch)

            # Expand dimensions - add batch dimension for the numpy
            resized_patch = np.expand_dims(resized_patch, axis=0)
            y_labels = np.expand_dims(y_labels, axis=0)

            patches.append(np.copy(resized_patch))
            labels.append(np.copy(y_labels))

        return np.concatenate(patches, axis=0), np.concatenate(labels, axis=0)

    def pre_process_predicates_data(self, image, ind, img):
        """
        This function is a generator for Predicate with Detections with batch-size
        :param image: image entity VG
        :param ind: index of an image of the whole images (used only for entities mixup Jitter)
        :return: This function will return numpy array of training images, numpy array of labels
        """
        patches = []
        labels = []
        url = image.image.url
        indices = set(range(self.size))

        if img is None:
            Logger().log("Couldn't get the image in url {}".format(url))
            return None, None

        # Create object pairs
        entities_pairs = list(itertools.product(image.objects, repeat=2))

        # Create a dict with key as pairs - (subject, object) and their values are predicates use for labels
        relations_dict = {}

        # Create a dict with key as pairs - (subject, object) and their values are relation index_id
        relations_filtered_id_dict = {}
        for relation in image.relationships:
            relations_dict[(relation.subject.id, relation.object.id)] = relation.predicate
            relations_filtered_id_dict[(relation.subject.id, relation.object.id)] = relation.filtered_id

        for relation in entities_pairs:

            subject = relation[0]
            object = relation[1]

            # Get the label of object
            if (subject.id, object.id) in relations_dict:
                label = relations_dict[(subject.id, object.id)]

            else:
                # Negative label
                label = "neg"

            # Check if it is a correct label
            if label not in self.hierarchy_mapping_predicates.keys():
                Logger().log("WARNING: label isn't familiar")
                return None

            # Get the label uuid
            label_id = self.hierarchy_mapping_predicates[label]

            # Create the y labels as a one hot vector
            y_labels = np.eye(len(self.hierarchy_mapping_predicates), dtype='uint8')[label_id]

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
            # FIXME resized_img = get_img_resize(patch_predicate, config.crop_width, config.crop_height,
            resized_patch = get_img_resize(patch_predicate, 112, 112,
                                           type=self.config.padding_method)
            resized_heatmap_subject = get_img_resize(patch_heatmap_heat_map_subject, 112, 112, type=self.config.padding_method)
            resized_heatmap_object = get_img_resize(patch_heatmap_heat_map_object, 112, 112, type=self.config.padding_method)

            # Augment only in training
            if self.is_train == 'train' and self.config.use_jitter:
                new_resized_patch = None

                # For mixup Jitter we need to create a new resize_img from another sample
                if False: #self.config.jitter.use_mixup:
                    all_indice_without_ind = list(indices - set([ind]))
                    # Pick different index from the data with no repetition
                    new_ind = np.random.choice(all_indice_without_ind)
                    new_object = self.train_images[new_ind]

                    new_img = get_img(new_object.url, download=True)
                    if new_img is None:
                        Logger().log("Coulden't get the image")
                        continue
                    # Get the mask: a dict with {x1,x2,y1,y2}
                    new_mask = get_mask_from_object(new_object)
                    # Cropping the patch from the image.
                    new_patch = new_img[new_mask['y1']: new_mask['y2'], new_mask['x1']: new_mask['x2'], :]
                    # Resize the image according the padding method
                    new_resized_patch = get_img_resize(new_patch, self.config.crop_width, self.config.crop_height,
                                                     type=self.config.padding_method)

                resized_patch = self.config.jitter.apply_jitter(resized_img=resized_patch, batchsize=self.size,
                                                                new_resized_img=new_resized_patch)

            # Concatenate the heat-map to the image in the kernel axis
            resized_patch = np.concatenate((resized_patch, resized_heatmap_subject[:, :, :1]), axis=2)
            resized_patch = np.concatenate((resized_patch, resized_heatmap_object[:, :, :1]), axis=2)

            # Expand dimensions - add batch dimension for the numpy
            resized_patch = np.expand_dims(resized_patch, axis=0)
            y_labels = np.expand_dims(y_labels, axis=0)

            patches.append(np.copy(resized_patch))
            labels.append(np.copy(y_labels))

        # slices sizes
        slices_size = np.zeros((3))
        slices_size[0] = slices_size[1] = len(entities_pairs) / 3
        slices_size[2] = len(entities_pairs) - slices_size[0] - slices_size[1]

        return np.concatenate(patches, axis=0), np.concatenate(labels, axis=0), slices_size

    def run(self):
        ind = 0
        for image in self.train_images:

            # filter images that fails in resize
            if image.image.id in [2379987, 2374549, 2351430, 2387196, 2403903, 2387505]:
                continue

            # filter non mixed cases
            relations_neg_labels = image.predicates_labels[:, :, NOF_PREDICATES - 1:]
            if np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                    relations_neg_labels) == 0:
                continue

            # filter images with more than 25 entities to avoid from OOM (just for train)
            if image.predicates_labels.shape[0] > 25:
                continue

            indices = np.arange(image.predicates_outputs_with_no_activation.shape[0])
            image.predicates_labels[indices, indices, :] = self.relation_neg

            # spatial features
            entity_bb = np.zeros((len(image.objects), 4))
            for obj_id in range(len(image.objects)):
                entity_bb[obj_id][0] = image.objects[obj_id].x / 1200.0
                entity_bb[obj_id][1] = image.objects[obj_id].y / 1200.0
                entity_bb[obj_id][2] = (image.objects[obj_id].x + image.objects[obj_id].width) / 1200.0
                entity_bb[obj_id][3] = (image.objects[obj_id].y + image.objects[obj_id].height) / 1200.0

            # Get image
            img_input = get_img(image.image.url, download=True)
            img_input = np.expand_dims(img_input, axis=0)

            entity_inputs, _ = self.pre_process_entities_data(image, ind, img_input)
            relations_inputs, _, slices_size = self.pre_process_predicates_data(image, ind, img_input)

            # give lower weight to negatives
            coeff_factor = np.ones(relations_neg_labels.shape)
            factor = float(np.sum(image.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                relations_neg_labels) / self.pred_pos_neg_ratio
            coeff_factor[relations_neg_labels == 1] *= factor

            coeff_factor[indices, indices] = 0

            # create the feed dictionary
            info = [image, relations_inputs, entity_inputs, entity_bb, slices_size, coeff_factor.reshape((-1)), indices,
                    img_input]

            self.queue.put(info)
            ind += 1

        self.queue.put(None)


def name_in_checkpoint(var):
    if "relation_resnet50" in var.op.name:
        return var.op.name.replace("relation_resnet50/", "")
    if "entity_resnet50" in var.op.name:
        return var.op.name.replace("entity_resnet50/", "")


def train(name="module",
          nof_iterations=100,
          learning_rate=0.0001,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          batch_size=20,
          pred_pos_neg_ratio=10,
          lr_object_coeff=1,
          layers=[500, 500, 500],
          gpu=0):
    """
    Train SGP module given train parameters and module hyper-parameters
    :param name: name of the train session
    :param nof_iterations: number of epochs
    :param learning_rate:
    :param learning_rate_steps: decay after number of steps
    :param learning_rate_decay: the factor to decay the learning rate
    :param load_module_name: name of already trained module weights to load
    :param use_saved_module: start from already train module
    :param batch_size: number of images in each mini-batch
    :param pred_pos_neg_ratio: Set the loss ratio between positive and negatives (not labeled) predicates
    :param lr_object_coeff: Set the loss ratio between objects and predicates
    :param layers: list of sizes of the hidden layer of the predicate and object classifier
    :param gpu: gpu number to use for the training
    :return: nothing
    """
    gpi_type = "FeatureAttention"
    including_object = True
    # get filesmanager
    filesmanager = FilesManager()

    # create logger
    logger_path = filesmanager.get_file_path("logs")
    logger_path = os.path.join(logger_path, name)
    logger = Logger(name, logger_path)

    # print train params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.log('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        logger.log("    %s = %s" % (i, values[i]))

    # set gpu
    if gpu != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # Load class config
    config = Config(gpu)
    # Print to the logger the config params
    config.config_logger()

    # create module
    module = End2EndModel(gpi_type=gpi_type, nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                          is_train=True,
                          learning_rate=learning_rate, learning_rate_steps=learning_rate_steps,
                          learning_rate_decay=learning_rate_decay,
                          lr_object_coeff=lr_object_coeff,
                          including_object=including_object,
                          layers=layers, config=config)

    # Get timestamp
    timestamp = get_time_and_date()

    # get module place holders
    #
    # get input place holders
    confidence_entity_ph, bb_ph = module.get_in_ph()
    # get labels place holders
    labels_relation_ph, labels_entity_ph, labels_coeff_loss_ph = module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = module.get_module_loss()

    ##
    # get module output
    out_relation_probes, out_entity_probes = module.get_output()

    # Initialize the Computational Graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Restore variables from disk.
        module_path = filesmanager.get_file_path("sg_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index") and use_saved_module:
            sess.run(init)
            # Add ops to save and restore all the variables.
            variables = tf.contrib.slim.get_variables_to_restore()
            variables_to_restore = [var for var in variables if not "ent_direct" in var.op.name and not "rel_direct" in var.op.name]
            saver = tf.train.Saver(variables_to_restore)

            saver.restore(sess, module_path_load)
            logger.log("Model restored.")
        else:
            create_folder(os.path.join(module_path, timestamp))
            sess.run(init)

            # # Load Object CNN body keras network
            # model_obj = tf.contrib.keras.models.Model(inputs=module.entity_inputs_ph,
            #                                           outputs=module.output_resnet50_entity_reshaped,
            #                                           name='entity_resnet50')
            # # The path for for loading Keras weights
            # net_weights = "/home/roeih/SceneGrapher/objects_no_fcs.h5"
            # model_obj.load_weights(net_weights, by_name=True)

            # Load Predicates MaskCNN keras network
            model_rel = tf.contrib.keras.models.Model(inputs=module.relation_inputs_ph, outputs=module.output_resnet50_relation,
                                                  name='relation_resnet50')
            # The path for for loading Keras weights
            net_weights = "/home/roeih/SceneGrapher/relations_no_fcs.h5"
            model_rel.load_weights(net_weights, by_name=True)

            # Save graph
            saver = tf.train.Saver()
            module_path_load = os.path.join(module_path, timestamp)
            saver.save(sess, module_path_load + '/e2e_fpn_model.ckpt', 0)

            # sess.run(init)
            # variables_to_restore = []
            # variables_to_restore += slim.get_model_variables("relation_resnet50")
            # variables_to_restore += slim.get_model_variables("entity_resnet50")
            # variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if "biases" not in var.op.name and "resnet_v2_50/logits/weights" not in var.op.name}
            # module_path = filesmanager.get_file_path("sg_module.train.saver")
            # module_path_load = os.path.join(module_path, "resnet_v2_50.ckpt")
            # #print_tensors_in_checkpoint_file(module_path_load, "", True)
            # #exit()
            # restorer = tf.train.Saver(variables_to_restore)
            # restorer.restore(sess, module_path_load)

        # train images
        vg_train_path = filesmanager.get_file_path("data.visual_genome.train")
        # list of train files
        # todo: debug
        # train_files_list = range(2, 72)
        train_files_list = range(0, 1)
        # shuffle(train_files_list)

        # Actual validation is 5 files.
        # After tunning the hyper parameters, use just 2 files for early stopping.
        validation_files_list = range(2)

        # create one hot vector for predicate_negative (i.e. not labeled)
        relation_neg = np.zeros(NOF_PREDICATES)
        relation_neg[NOF_PREDICATES - 1] = 1

        # object embedding
        # embed_obj = FilesManager().load_file("language_module.word2vec.object_embeddings")
        # embed_pred = FilesManager().load_file("language_module.word2vec.predicate_embeddings")
        # embed_pred = np.concatenate((embed_pred, np.zeros(embed_pred[:1].shape)),
        #                            axis=0)  # concat negative representation

        hierarchy_mapping_predicates = FilesManager().load_file("data.visual_genome.hierarchy_mapping_predicates")
        hierarchy_mapping_objects = FilesManager().load_file("data.visual_genome.hierarchy_mapping_objects")

        # train module
        lr = learning_rate
        best_test_loss = -1
        for epoch in xrange(1, nof_iterations):
            accum_results = None
            total_loss = 0
            steps = []
            # read data
            file_index = -1

            for file_name in train_files_list:
                file_index += 1
                # load data from file
                file_path = os.path.join(vg_train_path, str(file_name) + ".p")
                file_handle = open(file_path, "rb")
                train_images = cPickle.load(file_handle)
                file_handle.close()
                train_images = train_images[:1]
                pre_process_image_queue = Queue(maxsize=10)
                worker1 = PreProcessWorker(module=module, train_images=train_images,
                                           relation_neg=relation_neg, queue=pre_process_image_queue, lr=lr,
                                           pred_pos_neg_ratio=pred_pos_neg_ratio,
                                           hierarchy_mapping_objects=hierarchy_mapping_objects,
                                           hierarchy_mapping_predicates=hierarchy_mapping_predicates,
                                           config=config, is_train=True)
                # worker2 = PreProcessWorker(module=module, train_images=train_images[len(train_images) / 2:],
                #                            relation_neg=relation_neg,
                #                            queue=pre_process_image_queue, lr=lr,
                #                            pred_pos_neg_ratio=pred_pos_neg_ratio,
                #                            hierarchy_mapping_objects=hierarchy_mapping_objects,
                #                            hierarchy_mapping_predicates=hierarchy_mapping_predicates,
                #                            config=config, is_train=True)
                worker1.start()
                # worker2.start()
                none_count = 0
                while True:
                    # print(str(pre_process_image_queue.qsize()))
                    info = pre_process_image_queue.get()
                    if info is None:
                        none_count += 1
                        if none_count == 2:
                            break
                        continue

                    image = info[0]
                    relations_inputs = info[1]
                    entity_inputs = info[2]
                    entity_bb = info[3]
                    slices_size = info[4]
                    coeff_factor = info[5]
                    indices = info[6]
                    img_pixel = info[7]

                    feed_dict = {module.image_ph: img_pixel,
                                 module.relation_inputs_ph: relations_inputs,
                                 module.entity_inputs_ph: entity_inputs,
                                 module.num_objects_ph: (entity_inputs.shape[0],),
                                 module.entity_bb_ph: entity_bb, module.phase_ph: True,
                                 module.labels_relation_ph: image.predicates_labels,
                                 module.labels_entity_ph: image.objects_labels,
                                 module.labels_coeff_loss_ph: coeff_factor,
                                 module.lr_ph: lr}

                    # run the network
                    out_relation_probes_val, out_entity_probes_val, loss_val, gradients_val = \
                        sess.run([out_relation_probes, out_entity_probes, loss, gradients],
                                 feed_dict=feed_dict)

                    if math.isnan(loss_val):
                        print("NAN")
                        continue

                    # set diagonal to be neg (in order not to take into account in statistics)
                    out_relation_probes_val[indices, indices, :] = relation_neg

                    # append gradient to list (will be applied as a batch of entities)
                    steps.append(gradients_val)

                    # statistic
                    total_loss += loss_val

                    results = test(image.predicates_labels, image.objects_labels, out_relation_probes_val,
                                   out_entity_probes_val)

                    # accumulate results
                    if accum_results is None:
                        accum_results = results
                    else:
                        for key in results:
                            accum_results[key] += results[key]

                    if len(steps) == batch_size:
                        # apply steps
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
                # print stat - per file just for the first epoch
                if epoch == 1:
                    obj_accuracy = float(accum_results['entity_correct']) / accum_results['entity_total']
                    predicate_pos_accuracy = float(accum_results['relations_pos_correct']) / accum_results[
                        'relations_pos_total']
                    relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                        'relations_pos_total']
                    logger.log("iter %d.%d - obj %f - pred %f - relation %f" %
                               (epoch, file_index, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy))

            # print stat per epoch
            obj_accuracy = float(accum_results['entity_correct']) / accum_results['entity_total']
            predicate_pos_accuracy = float(accum_results['relations_pos_correct']) / accum_results[
                'relations_pos_total']
            predicate_all_accuracy = float(accum_results['relations_correct']) / accum_results['relations_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results[
                'relations_pos_total']
            relationships_all_accuracy = float(accum_results['relationships_correct']) / accum_results[
                'relations_total']

            logger.log("iter %d - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - lr %f" %
                       (epoch, total_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                        predicate_all_accuracy, relationships_all_accuracy, lr))

            # run validation
            if epoch % TEST_ITERATIONS == 0:
                total_test_loss = 0
                accum_test_results = None

                for file_name in validation_files_list:
                    # load data from file
                    file_path = os.path.join(vg_train_path, str(file_name) + ".p")
                    file_handle = open(file_path, "rb")
                    validation_images = cPickle.load(file_handle)
                    file_handle.close()

                    pre_process_image_queue = Queue(maxsize=10)
                    worker1 = PreProcessWorker(module=module, train_images=validation_images[:len(train_images) / 2],
                                               relation_neg=relation_neg, queue=pre_process_image_queue, lr=lr,
                                               pred_pos_neg_ratio=pred_pos_neg_ratio,
                                               hierarchy_mapping_objects=hierarchy_mapping_objects,
                                               hierarchy_mapping_predicates=hierarchy_mapping_predicates,
                                               config=config, is_train=False)
                    worker2 = PreProcessWorker(module=module, train_images=validation_images[len(train_images) / 2:],
                                               relation_neg=relation_neg,
                                               queue=pre_process_image_queue, lr=lr,
                                               pred_pos_neg_ratio=pred_pos_neg_ratio,
                                               hierarchy_mapping_objects=hierarchy_mapping_objects,
                                               hierarchy_mapping_predicates=hierarchy_mapping_predicates,
                                               config=config, is_train=False)
                    worker1.start()
                    worker2.start()
                    none_count = 0
                    while True:
                        # print(str(pre_process_image_queue.qsize()))
                        info = pre_process_image_queue.get()
                        if info is None:
                            none_count += 1
                            if none_count == 2:
                                break
                            continue

                        image = info[0]
                        relations_inputs = info[1]
                        entity_inputs = info[2]
                        entity_bb = info[3]
                        slices_size = info[4]
                        coeff_factor = info[5]
                        indices = info[6]

                        feed_dict = {module.relation_inputs_ph: relations_inputs,
                                     module.entity_inputs_ph: entity_inputs,
                                     module.num_objects_ph: (entity_inputs.shape[0],),
                                     module.entity_bb_ph: entity_bb, module.phase_ph: True,
                                     module.labels_relation_ph: image.predicates_labels,
                                     module.labels_entity_ph: image.objects_labels,
                                     module.labels_coeff_loss_ph: coeff_factor,
                                     module.lr_ph: lr}
                        # run the network
                        out_relation_probes_val, out_entity_probes_val, loss_val = sess.run(
                            [out_relation_probes, out_entity_probes, loss],
                            feed_dict=feed_dict)

                        # set diagonal to be neg (in order not to take into account in statistics)
                        out_relation_probes_val[indices, indices, :] = relation_neg

                        # statistic
                        total_test_loss += loss_val

                        # statistics
                        results = test(image.predicates_labels, image.objects_labels,
                                       out_relation_probes_val, out_entity_probes_val)

                        # accumulate results
                        if accum_test_results is None:
                            accum_test_results = results
                        else:
                            for key in results:
                                accum_test_results[key] += results[key]

                # print stat
                obj_accuracy = float(accum_test_results['entity_correct']) / accum_test_results['entity_total']
                predicate_pos_accuracy = float(accum_test_results['relations_pos_correct']) / accum_test_results[
                    'relations_pos_total']
                predicate_all_accuracy = float(accum_test_results['relations_correct']) / accum_test_results[
                    'relations_total']
                relationships_pos_accuracy = float(accum_test_results['relationships_pos_correct']) / \
                                             accum_test_results[
                                                 'relations_pos_total']
                relationships_all_accuracy = float(accum_test_results['relationships_correct']) / accum_test_results[
                    'relations_total']

                logger.log("VALIDATION - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f" %
                           (total_test_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy))

                # save best module so far
                if best_test_loss == -1 or total_test_loss < best_test_loss:
                    module_path_save = os.path.join(module_path, name + "_best_module.ckpt")
                    save_path = saver.save(sess, module_path_save)
                    logger.log("Model saved in file: %s" % save_path)
                    best_test_loss = total_test_loss

            # save module
            if epoch % SAVE_MODEL_ITERATIONS == 0:
                module_path_save = os.path.join(module_path, name + "_module.ckpt")
                save_path = saver.save(sess, module_path_save)
                logger.log("Model saved in file: %s" % save_path)

            # learning rate decay
            if (epoch % learning_rate_steps) == 0:
                lr *= learning_rate_decay


if __name__ == "__main__":
    filemanager = FilesManager()

    params = filemanager.load_file("e2e_module.train.params")

    name = params["name"]
    learning_rate = params["learning_rate"]
    learning_rate_steps = params["learning_rate_steps"]
    learning_rate_decay = params["learning_rate_decay"]
    nof_iterations = params["nof_iterations"]
    load_model_name = params["load_model_name"]
    use_saved_model = params["use_saved_model"]
    batch_size = params["batch_size"]
    predicate_pos_neg_ratio = params["predicate_pos_neg_ratio"]
    lr_object_coeff = params["lr_object_coeff"]
    layers = params["layers"]
    gpu = params["gpu"]

    train(name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
          use_saved_model, batch_size, predicate_pos_neg_ratio, lr_object_coeff, layers,
          gpu)
