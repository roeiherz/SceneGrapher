import sys

from LanguageTrain import pre_process_data

sys.path.append("..")
from LanguageModule import LanguageModule
import cPickle
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
import os
import inspect

__author__ = 'roeih'

# feature sizes
NUM_HIDDEN = 150
NUM_INPUT = 300
NOF_PREDICATES = 51
NOF_OBJECTS = 150
# apply gradients every batch size
BATCH_SIZE = 100
# negative vs positive factor
POS_NEG_RATIO = 0.3
# Use Predcls task
PREDCLS = True


def eval(nof_iterations=100,
         learning_rate=0.1,
         learning_rate_steps=1000,
         learning_rate_decay=0.5,
         load_module_name="module.ckpt",
         timesteps=1,
         gpu=0,
         files_train_list=None,
         files_test_list=None):
    """
    This function loads load_module_name and performs evaluation
    :param files_test_list: the list of test files
    :param files_train_list: the list of train files
    :param nof_iterations: num of iterations
    :param learning_rate: the lr
    :param learning_rate_steps: the num of steps which the lr will be updated
    :param learning_rate_decay: the decay of the lr
    :param load_module_name: the module file name
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

    Logger().log("    %s = %s" % ("PredCLS task", PREDCLS))

    # Create Module
    language_module = LanguageModule(timesteps=timesteps, is_train=True, num_hidden=NUM_HIDDEN,
                                     num_classes=NOF_PREDICATES, num_input=NUM_INPUT, learning_rate=learning_rate,
                                     learning_rate_steps=learning_rate_steps, learning_rate_decay=learning_rate_decay)

    # get input place holders
    inputs_ph = language_module.get_inputs_placeholders()
    # get labels place holders
    labels_ph = language_module.get_labels_placeholders()
    # get coeff place holders
    coeff_loss_ph = language_module.get_coeff_placeholders()
    # get learning rate place holder
    lr_ph = language_module.get_lr_placeholder()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = language_module.module_loss()
    # get module output
    accuracy = language_module.get_output()
    # get logits (likelihood)
    logits = language_module.get_logits()
    # get predictions (a softmax of the likelihood)
    softmax_predictions = language_module.get_predictions()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Define Summaries
    tf_logs_path = FilesManager().get_file_path("language_module.train.tf_logs")
    summary_writer = tf.summary.FileWriter(tf_logs_path, graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()

    Logger().log("Start Training")
    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = FilesManager().get_file_path("language_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index"):
            saver.restore(sess, module_path_load)
            Logger().log("Model restored.")
        else:
            Logger().log("Problem. No saver has been found in location {}".format(module_path_load + ".index"))
            return

        # Get the entities
        entities_path = FilesManager().get_file_path("data.visual_genome.detections_v4")
        files_test_list = ["Sat_Nov_11_21:59:10_2017"]

        if files_train_list is None or len(files_train_list) == 0:
            Logger().log("Error: No training data")
            return None

        if files_test_list is None or len(files_test_list) == 0:
            Logger().log("Error: No testing data")
            return None

        # Load hierarchy_mappings
        hierarchy_mapping_objects = FilesManager().load_file("data.visual_genome.hierarchy_mapping_objects")
        hierarchy_mapping_predicates = FilesManager().load_file("data.visual_genome.hierarchy_mapping_predicates")
        # Load pre-trained objects embeddings
        objects_embeddings = FilesManager().load_file("language_module.word2vec.object_embeddings")

        # Create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1

        # module
        for epoch in range(1, nof_iterations):
            try:
                # region Testing
                # read data
                test_total_acc = 0
                test_total_loss = 0
                test_num_entities = 0
                # test_num_predicates_total = 0

                for file_dir in files_test_list:
                    files = os.listdir(os.path.join(entities_path, file_dir))
                    for file_name in files:

                        # Load only entities
                        if ".log" in file_name or "lang" in file_name:
                            continue

                        file_path = os.path.join(entities_path, file_dir, file_name)
                        file_handle = open(file_path, "rb")
                        test_entities = cPickle.load(file_handle)
                        file_handle.close()
                        for entity in test_entities:
                            try:

                                if len(entity.relationships) == 0:
                                    continue

                                # Pre-processing entities to get RNN inputs and outputs
                                rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects,
                                                                           objects_embeddings,
                                                                           pos_neg_ratio=0.0,
                                                                           pred_cls=PREDCLS)

                                # Create the feed dictionary
                                feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs}

                                # Run the network
                                accuracy_val, loss_val, logits_val, softmax_predictions_val = sess.run(
                                    [accuracy, loss, logits, softmax_predictions],
                                    feed_dict=feed_dict)

                                # Calculates loss
                                test_total_loss += loss_val
                                # Calculates accuracy
                                test_total_acc += accuracy_val
                                # Append number of predicates
                                # test_num_predicates_total += rnn_outputs.shape[0]
                                # Update the number of entities
                                test_num_entities += 1

                            except Exception as e:
                                logger.log(
                                    "Error: problem in Test. Epoch: {0}, image id: {1} Exception: {2}".format(epoch,
                                                                                                              entity.image.id,
                                                                                                              str(
                                                                                                                  e)))
                                continue

                # Print stats
                Logger().log("TEST EPOCH: epoch: %d - loss: %f - predicates accuracy: %f " %
                             (epoch, float(test_total_loss) / test_num_entities,
                              float(test_total_acc) / test_num_entities))

            # endregion
            # Finished Testing

            except Exception as e:
                logger.log("Error: problem in epoch: {0} with: {1}".format(epoch, str(e)))
                continue


if __name__ == "__main__":
    filemanager = FilesManager()
    logger = Logger()

    params = filemanager.load_file("language_module.train.params")
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
        timesteps = process_params["timesteps"]
        gpu = process_params["gpu"]
        files_train_list = process_params["files_train"]
        files_test_list = process_params["files_test"]

        eval(nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
             timesteps, gpu, files_train_list, files_test_list)

        # p = Process(target=train, args=(
        #     name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
        #     use_saved_model, timesteps, gpu))
        # p.start()
        # processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
