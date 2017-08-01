import inspect
import sys
sys.path.append("..")
from multiprocessing import Process

from FilesManager.FilesManager import FilesManager
from Module import Module
import tensorflow as tf
import numpy as np
import os

from Utils.Logger import Logger

VISUAL_FEATURES_PREDICATE_SIZE = 2048
# VISUAL_FEATURES_PREDICATE_SIZE = 2
VISUAL_FEATURES_OBJECT_SIZE = 2048
# VISUAL_FEATURES_OBJECT_SIZE = 2
NOF_PREDICATES = 51
# NOF_PREDICATES = 2
NOF_OBJECTS = 150


# NOF_OBJECTS = 2



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

    # number of object correct predictipns
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
    predicats_pred_true = predicats_pred[object_true_indices[0],:][:, object_true_indices[0]]    
    results["relationships_correct"] = np.sum(predicats_gt_true == predicats_pred_true)
    # number of correct postive relationships
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
    labels_predicate_ph, labels_object_ph = module.get_labels_ph()
    # get loss and train step
    loss, train_step = module.get_module_loss()
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
            saver.restore(sess, module_path)
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


        # read data
        entities = filesmanager.load_file("data.visual_genome.detections_v2")

        # train module
        for epoch in range(nof_iterations):
            accum_results = None
            total_loss = 0
            for entity in entities:
                # get shape of extended object to be used by the module
                extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                extended_belief_object_shape[2] = NOF_OBJECTS
                # give lower weight to negatives
                predicates_labels = entity.predicates_labels.astype(float)
		predicates_labels[:, :,:NOF_PREDICATES-2] = 0
                factor = float(np.sum(entity.predicates_labels[:, :,:NOF_PREDICATES-2])) / np.sum(entity.predicates_labels) / 3 - 1
                #print(factor)
                predicates_labels *= factor
		#predicates_labels *= - 0.99
                predicates_labels += entity.predicates_labels		
                # create the feed dictionary
                feed_dict = {belief_predicate_ph: entity.predicates_probes, belief_object_ph: entity.objects_probs,
                             extended_belief_object_shape_ph: extended_belief_object_shape,
                             visual_features_predicate_ph: entity.predicates_features,
                             visual_features_object_ph: entity.objects_features,
                             labels_predicate_ph: predicates_labels, labels_object_ph: entity.objects_labels}

                out_belief_predicate_val, out_belief_object_val, loss_val, train_step_val, lr = \
                    sess.run([out_belief_predicate, out_belief_object, loss, train_step, module.learning_rate_var],
                             feed_dict=feed_dict)
                
                total_loss += loss_val

                results = test(entity.predicates_labels, entity.objects_labels, out_belief_predicate_val, out_belief_object_val)
                #results = test(entity.predicates_labels, entity.objects_labels, entity.predicates_probes, entity.objects_probs)
                # accumulate results
                if accum_results is None:
                    accum_results = results
                else:
                    for key in results:
                        accum_results[key] += results[key]
        
            # print stat
            obj_accuracy = float(accum_results['obj_correct']) / accum_results['obj_total']
            predicate_pos_accuracy = float(accum_results['predicates_pos_correct']) / accum_results['predicates_pos_total']
            predicate_all_accuracy = float(accum_results['predicates_correct']) / accum_results['predicates_total']
            relationships_pos_accuracy = float(accum_results['relationships_pos_correct']) / accum_results['predicates_pos_total']
            relationships_all_accuracy = float(accum_results['relationships_correct']) / accum_results['predicates_total']


            logger.log("iter %d - loss %f - obj %f - pred %f - rela %f - all_pred %f - all rela %f - lr %f" %
                       (epoch, total_loss, obj_accuracy, predicate_pos_accuracy, relationships_pos_accuracy,
                        predicate_all_accuracy, relationships_all_accuracy, lr))
            if epoch % 100 == 0:
                module_path_save = os.path.join(module_path, name + "_module.ckpt")
                save_path = saver.save(sess, module_path_save)
                logger.log("Model saved in file: %s" % save_path) 
        print("Debug")

        # save module
        module_path_save = os.path.join(module_path, name + "_module.ckpt")
        save_path = saver.save(sess, module_path_save)
        logger.log("Model saved in file: %s" % save_path)

    print("Debug")


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
        p = Process(target=train, args=(
            name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
            use_saved_model, gpu))
        p.start()
        processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
