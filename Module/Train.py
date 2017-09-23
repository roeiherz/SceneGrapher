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

# feature sizes
VISUAL_FEATURES_PREDICATE_SIZE = 2048
VISUAL_FEATURES_OBJECT_SIZE = 2048
NOF_PREDICATES = 51
NOF_OBJECTS = 150

# negative vs positive factor
POS_NEG_FACTOR = 3.3

# save model every number of iterations
SAVE_MODEL_ITERATIONS = 5

# test every number of iterations
TEST_ITERATIONS = 1

# percentage of test data
TEST_PERCENT = 10

# apply gradients every batch size
BATCH_SIZE = 100


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
          rnn_steps=1,
          loss_func="all",
          lr_object_coeff=1,
          including_object=False,
          gpu=0):

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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.log("os.environ[\"CUDA_VISIBLE_DEVICES\"] = " + str(gpu))

    # create module
    module = Module(nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                    visual_features_predicate_size=VISUAL_FEATURES_PREDICATE_SIZE,
                    visual_features_object_size=VISUAL_FEATURES_OBJECT_SIZE, is_train=True,
                    learning_rate=learning_rate, learning_rate_steps=learning_rate_steps,
                    learning_rate_decay=learning_rate_decay,
                    rnn_steps=rnn_steps,
                    loss_func=loss_func,
                    lr_object_coeff=lr_object_coeff,
                    including_object=including_object)

    # get input place holders
    belief_predicate_ph, belief_object_ph, extended_belief_object_shape_ph, visual_features_predicate_ph, visual_features_object_ph = module.get_in_ph()
    # get labels place holders
    labels_predicate_ph, labels_object_ph, labels_coeff_loss_ph = module.get_labels_ph()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = module.get_module_loss()
    # get module output
    out_predicate_probes, out_object_probes = module.get_output()

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

         # get object labels to ids and predicate labels to ids
        _, object_ids, predicate_ids = get_filtered_data(filtered_data_file_name="mini_filtered_data", category='entities_visual_module')

        # train entities
        entities_path = filesmanager.get_file_path("data.visual_genome.detections_v4")
        files_list = ["Wed_Aug_23_14:01:18_2017/predicated_entities_0_to_1000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_1000_to_2000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_2000_to_3000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_3000_to_4000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_4000_to_5000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_5000_to_6000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_6000_to_6500.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_6500_to_7000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_7000_to_7500.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_8000_to_8500.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_8500_to_9000.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_9000_to_9500.p", "Wed_Aug_23_14:01:18_2017/predicated_entities_9500_to_10000.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_0_to_1000.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_1000_to_2000.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_2000_to_3000.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_3000_to_4000.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_4000_to_5000.p",  "Wed_Aug_23_14:00:45_2017/predicated_entities_5000_to_5500.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_5500_to_6000.p",  "Wed_Aug_23_14:00:45_2017/predicated_entities_6000_to_6500.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_6500_to_7000.p",  "Wed_Aug_23_14:00:45_2017/predicated_entities_7000_to_7500.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_7500_to_8000.p",  "Wed_Aug_23_14:00:45_2017/predicated_entities_8000_to_8500.p", "Wed_Aug_23_14:00:45_2017/predicated_entities_8500_to_9000.p",  "Wed_Aug_23_14:00:45_2017/predicated_entities_9000_to_9500.p"]
        # get mini data to filter
        img_ids = Scripts.get_img_ids()
        # read test entities
        test_entities = filesmanager.load_file("data.visual_genome.detections_v4_test")

        # create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1

        # train module
        lr = learning_rate
        best_test_loss = -1
        for epoch in range(1, nof_iterations):
            accum_results = None
            total_loss = 0
            steps = []
            # read data
            for file_name in files_list:
                file_path = os.path.join(entities_path, file_name)
                file_handle = open(file_path, "rb")
                train_entities = cPickle.load(file_handle)
                file_handle.close()
                for entity in train_entities:
                    # filter mini data urls to be used as test urls
                    if entity.image.id in img_ids:
                        continue

                    # set diagonal to be neg
                    indices = np.arange(entity.predicates_probes.shape[0])
                    entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                    entity.predicates_labels[indices, indices, :] = predicate_neg
                    entity.predicates_probes[indices, indices, :] = predicate_neg

                    # get shape of extended object to be used by the module
                    extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                    extended_belief_object_shape[2] = NOF_OBJECTS

                    # filter non mixed cases
                    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES-1:]
                    if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(predicates_neg_labels) == 0:
                       continue

                    
                    #in_object_belief = entity.objects_labels * 10
                    in_object_belief = entity.objects_outputs_with_no_activations
                    # give lower weight to negatives
                    coeff_factor = np.ones(predicates_neg_labels.shape)
                    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                        predicates_neg_labels) / POS_NEG_FACTOR 
                    coeff_factor[predicates_neg_labels == 1] *= factor
                    coeff_factor[indices, indices] = 0

                    # create the feed dictionary
                    feed_dict = {belief_predicate_ph: entity.predicates_outputs_with_no_activation, belief_object_ph: in_object_belief,
                                 extended_belief_object_shape_ph: extended_belief_object_shape,
                                 visual_features_predicate_ph: entity.predicates_features,
                                 visual_features_object_ph: entity.objects_features,
                                 labels_predicate_ph: entity.predicates_labels, labels_object_ph: entity.objects_labels, labels_coeff_loss_ph: coeff_factor.reshape((-1)),  module.lr_ph : lr}

                    # run the network
                    out_predicate_probes_val, out_object_probes_val, loss_val, gradients_val = \
                        sess.run([out_predicate_probes, out_object_probes, loss, gradients],
                                 feed_dict=feed_dict)

                    # set diagonal to be neg (in order not to take into account in statistics)
                    out_predicate_probes_val[indices, indices, :] = predicate_neg

                    # append gradient to list (will be applied as a batch of entities)
                    steps.append(gradients_val)

                    # statistic
                    total_loss += loss_val
                    results = test(entity.predicates_labels, entity.objects_labels, out_predicate_probes_val,
                                   out_object_probes_val)

                    # accumulate results
                    if accum_results is None:
                        accum_results = results
                    else:
                        for key in results:
                            accum_results[key] += results[key]

                    if len(steps) == BATCH_SIZE:
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
                total_test_loss = 0
                accum_test_results = None
                correct_predicate = 0
                total_predicate = 0

                for entity in test_entities:

                    # set diagonal to be neg
                    indices = np.arange(entity.predicates_probes.shape[0])
                    entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                    entity.predicates_labels[indices, indices, :] = predicate_neg
                    entity.predicates_probes[indices, indices, :] = predicate_neg

                    # get shape of extended object to be used by the module
                    extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                    extended_belief_object_shape[2] = NOF_OBJECTS

                    # filter non mixed cases
                    predicates_neg_labels = entity.predicates_labels[:, :, NOF_PREDICATES-1:]
                    if np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2]) == 0 or np.sum(predicates_neg_labels) == 0:
                       continue
                    
                    # give lower weight to negatives
                    coeff_factor = np.ones(predicates_neg_labels.shape)
                    factor = float(np.sum(entity.predicates_labels[:, :, :NOF_PREDICATES - 2])) / np.sum(
                        predicates_neg_labels) / POS_NEG_FACTOR 
                    coeff_factor[predicates_neg_labels == 1] *= factor
                    coeff_factor[indices, indices] = 0
                    
                    #in_object_belief = entity.objects_labels * 10
                    in_object_belief = entity.objects_outputs_with_no_activations
                    
                    # create the feed dictionary
                    feed_dict = {belief_predicate_ph: entity.predicates_outputs_with_no_activation, belief_object_ph: in_object_belief,
                                 extended_belief_object_shape_ph: extended_belief_object_shape,
                                 visual_features_predicate_ph: entity.predicates_features,
                                 visual_features_object_ph: entity.objects_features,
                                 labels_predicate_ph: entity.predicates_labels, labels_object_ph: entity.objects_labels,
labels_coeff_loss_ph: coeff_factor.reshape((-1))}

                    # run the network
                    out_predicate_probes_val, out_object_probes_val, loss_val = sess.run(
                        [out_predicate_probes, out_object_probes, loss],
                        feed_dict=feed_dict)

                    # set diagonal to be neg (in order not to take into account in statistics)
                    out_predicate_probes_val[indices, indices, :] = predicate_neg

                    # statistic
                    total_test_loss += loss_val

                    # statistics
                    results = test(entity.predicates_labels, entity.objects_labels,
                                   out_predicate_probes_val, out_object_probes_val)

                    # accumulate results
                    if accum_test_results is None:
                        accum_test_results = results
                    else:
                        for key in results:
                            accum_test_results[key] += results[key]
                    
                    # eval per predicate
                    correct_predicate_image, total_predicate_image = predicate_class_recall(entity.predicates_labels,
                                                                                            out_predicate_probes_val)
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

                logger.log("TEST - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - top5 %f" %
                           (total_test_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                            predicate_all_accuracy, relationships_all_accuracy, float(correct_predicate)/total_predicate))

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
        rnn_steps = process_params["rnn_steps"]
        loss_func = process_params["loss_func"]
        lr_object_coeff = process_params["lr_object_coeff"]
        including_object = process_params["including_object"]
        gpu = process_params["gpu"]

        train(name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name, use_saved_model, rnn_steps, loss_func, lr_object_coeff, including_object, gpu)
        p = Process(target=train, args=(
            name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
            use_saved_model, rnn_steps, loss_func, lr_object_coeff, gpu))
        p.start()
        processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
