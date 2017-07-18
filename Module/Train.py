from Module import Module
import tensorflow as tf
import numpy as np
import os

VISUAL_FEATURES_PREDICATE_SIZE = 2048
# VISUAL_FEATURES_PREDICATE_SIZE = 2
VISUAL_FEATURES_OBJECT_SIZE = 2048
# VISUAL_FEATURES_OBJECT_SIZE = 2
NOF_PREDICATES = 51
# NOF_PREDICATES = 2
NOF_OBJECTS = 150
# NOF_OBJECTS = 2
NOF_ITERATIONS = 100


LOAD_MODEL_NAME = "module.ckpt"
USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "module.ckpt"


if __name__ == "__main__":
    # create module
    module = Module(nof_predicates=NOF_PREDICATES, nof_objects=NOF_OBJECTS,
                    visual_features_predicate_size=VISUAL_FEATURES_PREDICATE_SIZE,
                    visual_features_object_size=VISUAL_FEATURES_OBJECT_SIZE, is_train=True)

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
    summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        # Restore variables from disk.
        if os.path.exists(LOAD_MODEL_NAME + ".index") and USE_SAVED_MODEL:
            saver.restore(sess, "./" + LOAD_MODEL_NAME)
            print("Model restored.")
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
        for i in range(NOF_ITERATIONS):
            out_belief_predicate_val, out_belief_object_val, loss_val, train_step_val = \
                sess.run([out_belief_predicate, out_belief_object, loss, train_step],
                    feed_dict=feed_dict)

            print("iter %d - loss %f" % (i, loss_val))

        print("Debug")

        # save module
        save_path = saver.save(sess, SAVE_MODEL_NAME)
        print("Model saved in file: %s" % save_path)

    print("Debug")
