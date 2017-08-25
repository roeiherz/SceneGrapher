import inspect
import sys

sys.path.append("..")
from multiprocessing import Process

from FilesManager.FilesManager import FilesManager
from Module import Module
import tensorflow as tf
import numpy as np
import os
import cPickle
import Scripts
from FeaturesExtraction.Utils.data import get_filtered_data

from Utils.Logger import Logger
from Utils.Utils import preprocess_features, normalize_adj, preprocess_adj

VISUAL_FEATURES_PREDICATE_SIZE = 2048
# VISUAL_FEATURES_PREDICATE_SIZE = 2
VISUAL_FEATURES_OBJECT_SIZE = 2048
# VISUAL_FEATURES_OBJECT_SIZE = 2
NOF_PREDICATES = 51
# NOF_PREDICATES = 2
NOF_OBJECTS = 150
# NOF_OBJECTS = 2

# negative vs positive factor
POS_NEG_FACTOR = 10

# save model every number of iterations
SAVE_MODEL_ITERATIONS = 20

# test every number of iterations
TEST_ITERATIONS = 10

# percentage of test data
TEST_PERCENT = 10


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
    predicats_pred_pos = predicats_pred[pos_indices]
    results["predicates_pos_correct"] = np.sum(predicats_gt_pos == predicats_pred_pos)
    # number of correct relationships
    object_true_indices = np.where(objects_gt == objects_pred)
    predicats_gt_true = predicats_gt[object_true_indices[0], :][:, object_true_indices[0]]
    predicats_pred_true = predicats_pred[object_true_indices[0], :][:, object_true_indices[0]]
    results["relationships_correct"] = np.sum(predicats_gt_true == predicats_pred_true)
    # number of correct positive relationships
    pos_true_indices = np.where(predicats_gt_true != NOF_PREDICATES - 1)
    predicats_gt_pos_true = predicats_gt_true[pos_true_indices]
    predicats_pred_pos_true = predicats_pred_true[pos_true_indices]
    results["relationships_pos_correct"] = np.sum(predicats_gt_pos_true == predicats_pred_pos_true)

    return results


def train(name="test",
          nof_iterations=100,
          learning_rate=0.1,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          gpu=0):
    filesmanager = FilesManager()
    logger_path = filesmanager.get_file_path("logs")
    logger_path = os.path.join(logger_path, name)
    # create logger
    logger = Logger(name, logger_path)

    # print train params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.log('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        logger.log("    %s = %s" % (i, values[i]))

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # create module
    module = Module(nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                    visual_features_predicate_size=VISUAL_FEATURES_PREDICATE_SIZE,
                    visual_features_object_size=VISUAL_FEATURES_OBJECT_SIZE, is_train=True,
                    learning_rate=learning_rate, learning_rate_steps=learning_rate_steps,
                    learning_rate_decay=learning_rate_decay)

    # get input place holders
    belief_predicate_ph, belief_object_ph, extended_belief_object_shape_ph, visual_features_predicate_ph, visual_features_object_ph = module.get_in_ph()
    # get labels place holders
    labels_predicate_ph, labels_object_ph, labels_coeff_loss_ph = module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = module.get_module_loss()
    # get module output
    out_belief_predicate, out_belief_object = module.get_output()

    # Initialize the Computational Graph
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Define Summaries
    tf_logs_path = filesmanager.get_file_path("sg_module.train.tf_logs")
    summary_writer = tf.summary.FileWriter(tf_logs_path, graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = filesmanager.get_file_path("sg_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index") and use_saved_module:
            saver.restore(sess, module_path_load)
            logger.log("Model restored.")
        else:
            sess.run(init)

        # fake data to test
        # N = 3
        # belief_predicate = np.zeros((N, N, NOF_PREDICATES))
        # belief_object = np.zeros((N, NOF_OBJECTS))
        # extended_belief_object_shape = np.asarray(belief_predicate.shape)
        # extended_belief_object_shape[2] = NOF_OBJECTS
        # visual_features_predicate = np.arange(2000, 2000 + N * N * VISUAL_FEATURES_PREDICATE_SIZE).reshape(N, N,
        #                                                                                                    VISUAL_FEATURES_PREDICATE_SIZE)
        # visual_features_object = np.arange(3000, 3000 + N * VISUAL_FEATURES_OBJECT_SIZE).reshape(N,
        #                                                                                          VISUAL_FEATURES_OBJECT_SIZE)
        # labels_predicate = np.zeros((N, N, NOF_PREDICATES))
        # for i in range(N):
        #     for j in range(N):
        #         labels_predicate[i][j][i * j] = 1
        # labels_object = np.zeros((N, NOF_OBJECTS))
        # for i in range(N):
        #     labels_object[i][i] = 1
        _, object_ids, predicate_ids = get_filtered_data(filtered_data_file_name="mini_filtered_data",
                                                         category='entities_visual_module')
        # train entities
        entities_path = filesmanager.get_file_path("data.visual_genome.detections_v4")
        # FIXME: modified to have a single iteration
        # files_list = ["1"]
        files_list = ["Wed_Aug__9_10:04:43_2017/predicated_entities_0_to_1000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_1000_to_2000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_2000_to_3000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_3000_to_4000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_4000_to_5000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_0_to_1000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_1000_to_2000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_2000_to_3000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_3000_to_4000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_4000_to_5000.p"]
        img_ids = Scripts.get_img_ids()
        # read test entities
        test_entities = filesmanager.load_file("data.visual_genome.detections_v4_test")
        # create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1
        # train module
        lr = learning_rate

        for epoch in range(1, nof_iterations):
            accum_results = None
            total_loss = 0
            steps = []

            # read data
            for file_name in files_list:

                # FIXME: use test entities (mini-data) at first
                # train_entities = filesmanager.load_file("data.visual_genome.detections_v4_test")
                file_path = os.path.join(entities_path, file_name)
                file_handle = open(file_path, "rb")
                train_entities = cPickle.load(file_handle)
                file_handle.close()

                for entity in train_entities:

                    # filter mini data urls to be used as test urls
                    # FIXME: allow to train on mini-data
                    if entity.image.id in img_ids:
                        continue

                    # FIXME: filter data with errors
                    #object_gt = np.argmax(entity.objects_labels, axis=1)
                    #count_ids = np.bincount(object_gt)
                    #if np.max(count_ids) > 1:
                    #    continue

                    # set diagonal to be neg
                    indices = np.arange(entity.predicates_probes.shape[0])
                    entity.predicates_probes[indices, indices, :] = predicate_neg

                    # get shape of extended object to be used by the module
                    extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                    extended_belief_object_shape[2] = NOF_OBJECTS

                    # filter non mixed cases
                    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES - 1:]

                    if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(
                            predicates_neg_labels) == 0:
                        Logger().log("No Positive Predicate in entity_id {}".format(entity.image.id))

                    # give lower weight to negatives
                    coeff_factor = np.ones(predicates_neg_labels.shape)
                    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                        predicates_neg_labels) / POS_NEG_FACTOR
                    coeff_factor[predicates_neg_labels == 1] *= factor

                    # FIXME: train on true predicates only
                    coeff_factor[predicates_neg_labels == 1] = 0

                    # preprocessed_features(entity)

                    # create the feed dictionary
                    feed_dict = {belief_predicate_ph: entity.predicates_probes, belief_object_ph: entity.objects_probs,
                                 extended_belief_object_shape_ph: extended_belief_object_shape,
                                 visual_features_predicate_ph: entity.predicates_features,
                                 visual_features_object_ph: entity.objects_features,
                                 labels_predicate_ph: entity.predicates_labels, labels_object_ph: entity.objects_labels, labels_coeff_loss_ph: coeff_factor.reshape((-1)),  module.lr_ph : lr}

                    # run the network
                    out_belief_predicate_val, out_belief_object_val, loss_val, gradients_val = \
                        sess.run([out_belief_predicate, out_belief_object, loss, gradients],
                                 feed_dict=feed_dict)

                    calc_positive_accuracy(entity, out_belief_predicate_val)

                    # Store Gradients
                    steps.append(gradients_val)

                    # statistic
                    total_loss += loss_val
                    results = test(entity.predicates_labels, entity.objects_labels, out_belief_predicate_val,
                                   out_belief_object_val)
                    # results = test(entity.predicates_labels, entity.objects_labels, entity.predicates_probes, entity.objects_probs)
                    # accumulate results
                    if accum_results is None:
                        accum_results = results
                    else:
                        for key in results:
                            accum_results[key] += results[key]
                    if len(steps) == 10:
                        # apply steps
                        for step in steps:
                            feed_grad_apply_dict = {grad_placeholder[j][0]: step[j][0] for j in xrange(len(grad_placeholder))}
                            feed_grad_apply_dict[module.lr_ph] = lr
                            sess.run([train_step], feed_dict=feed_grad_apply_dict)
                        steps = []

            # print stat
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

            if epoch % TEST_ITERATIONS == 0:
                total_loss = 0
                accum_test_results = None
                correct_predicate = 0
                total_predicate = 0

                for entity in test_entities:
                    # FIXME: filter data with errors
                    #object_gt = np.argmax(entity.objects_labels, axis=1)
                    #count_ids = np.bincount(object_gt)
                    #if np.max(count_ids) > 1:
                    #    continue

                    # get shape of extended object to be used by the module
                    extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                    extended_belief_object_shape[2] = NOF_OBJECTS

                    # create the feed dictionary
                    feed_dict = {belief_predicate_ph: entity.predicates_probes, belief_object_ph: entity.objects_probs,
                                 extended_belief_object_shape_ph: extended_belief_object_shape,
                                 visual_features_predicate_ph: entity.predicates_features,
                                 visual_features_object_ph: entity.objects_features,
                                 labels_predicate_ph: entity.predicates_labels, labels_object_ph: entity.objects_labels}

                    # run the network
                    out_belief_predicate_val, out_belief_object_val = sess.run(
                        [out_belief_predicate, out_belief_object],
                        feed_dict=feed_dict)

                    # statistic
                    results = test(entity.predicates_labels, entity.objects_labels,
                                   out_belief_predicate_val, out_belief_object_val)
                    # accumulate results
                    if accum_test_results is None:
                        accum_test_results = results
                    else:
                        for key in results:
                            accum_test_results[key] += results[key]

                    # eval per predicate
                    correct_predicate_image, total_predicate_image = predicate_class_recall(entity.predicates_labels,
                                                                                            out_belief_predicate_val)
                    correct_predicate += np.sum(correct_predicate_image[:NOF_PREDICATES-2])
                    total_predicate += np.sum(total_predicate_image[:NOF_PREDICATES-2])

                # print stat
                obj_accuracy = float(accum_test_results['obj_correct']) / accum_test_results['obj_total']
                predicate_pos_accuracy = float(accum_test_results['predicates_pos_correct']) / accum_test_results[
                    'predicates_pos_total']
                predicate_all_accuracy = float(accum_test_results['predicates_correct']) / accum_test_results['predicates_total']
                relationships_pos_accuracy = float(accum_test_results['relationships_pos_correct']) / accum_test_results[
                    'predicates_pos_total']
                relationships_all_accuracy = float(accum_test_results['relationships_correct']) / accum_test_results[
                    'predicates_total']

                logger.log("TEST - obj %f - pred %f - rela %f - all_pred %f - all rela %f - top5 %f" %
                           (obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy, float(correct_predicate)/total_predicate))

            if epoch % SAVE_MODEL_ITERATIONS == 0:
                module_path_save = os.path.join(module_path, name + "_module.ckpt")
                save_path = saver.save(sess, module_path_save)
                logger.log("Model saved in file: %s" % save_path)
            if epoch % learning_rate_steps == 0:
                lr *= learning_rate_decay

        print("Debug")

        # save module
        module_path_save = os.path.join(module_path, name + "_module.ckpt")
        save_path = saver.save(sess, module_path_save)
        logger.log("Model saved in file: %s" % save_path)

    print("Debug")

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


def preprocessed_features(entity):
    # Find the adjacency matrix
    pos_labels_indices = np.where(entity.predicates_labels[:, :, 50] == 0)
    # Create a mask with zeros in negatives and 1 in positives
    mask = np.zeros(shape=entity.predicates_labels.shape)
    label_pos = \
        np.where(entity.predicates_labels[np.where(entity.predicates_labels[:, :, 50] == 0)] == 1)[1]
    mask[pos_labels_indices + (label_pos,)] = 1
    graph_adj = np.sum(mask, axis=2)
    normalized_graph_adj = preprocess_adj(graph_adj)
    predicates_features_avg = np.sum(entity.predicates_features, axis=2)
    predicates_preprocess_features = preprocess_features(predicates_features_avg)


def calc_positive_accuracy(entity, out_belief_predicate_val):
    pred = np.argmax(out_belief_predicate_val, axis=2)
    gt = np.argmax(entity.predicates_labels, axis=2)
    positives_preds = pred[np.where(gt != 50)]
    positives_gt = gt[np.where(gt != 50)]
    score = np.sum(positives_preds == positives_gt) / float(len(positives_preds))
    print(
        "In entity {0} the Score is {1} while number of positives is {2}".format(entity.image.id, score,
                                                                                 len(positives_preds)))


if __name__ == "__main__":
    filemanager = FilesManager()

    params = filemanager.load_file("sg_module.train.params")

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
        gpu = process_params["gpu"]
        train(name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
              use_saved_model, gpu)
        p = Process(target=train, args=(
            name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
            use_saved_model, gpu))
        p.start()
        processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
