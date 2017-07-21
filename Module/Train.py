import inspect
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
                    learning_rate=learning_rate, learning_rate_steps=learning_rate_steps, learning_rate_decay=learning_rate_decay)

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
        N = 3
        belief_predicate = np.arange(N*N*NOF_PREDICATES).reshape(N, N, NOF_PREDICATES)
        belief_object = np.arange(1000, 1000 + N * NOF_OBJECTS).reshape(N, NOF_OBJECTS)
        extended_belief_object_shape = np.asarray(belief_predicate.shape)
        extended_belief_object_shape[2] = NOF_OBJECTS
        visual_features_predicate = np.arange(2000, 2000 + N*N*VISUAL_FEATURES_PREDICATE_SIZE).reshape(N, N, VISUAL_FEATURES_PREDICATE_SIZE)
        visual_features_object = np.arange(3000, 3000 + N*VISUAL_FEATURES_OBJECT_SIZE).reshape(N, VISUAL_FEATURES_OBJECT_SIZE)
        labels_predicate = np.ones((N, N, NOF_PREDICATES))
        labels_object = np.ones((N, NOF_OBJECTS))
        feed_dict = {belief_predicate_ph: belief_predicate, belief_object_ph: belief_object, extended_belief_object_shape_ph: extended_belief_object_shape,
                     visual_features_predicate_ph: visual_features_predicate, visual_features_object_ph: visual_features_object,
                     labels_predicate_ph: labels_predicate, labels_object_ph: labels_object}

        # train module
        for i in range(nof_iterations):
            out_belief_predicate_val, out_belief_object_val, loss_val, train_step_val = \
                sess.run([out_belief_predicate, out_belief_object, loss, train_step],
                    feed_dict=feed_dict)

            logger.log("iter %d - loss %f" % (i, loss_val))

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
            name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name, use_saved_model, gpu))
        p.start()
        processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
