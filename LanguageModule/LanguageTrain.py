import sys
sys.path.append("..")
import csv
from FeaturesExtraction.Utils.Utils import get_time_and_date
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
NUM_HIDDEN = 150
NUM_INPUT = 300
NOF_PREDICATES = 51
NOF_OBJECTS = 150
# apply gradients every batch size
BATCH_SIZE = 100
# save model every number of iterations
SAVE_MODEL_ITERATIONS = 5
# test every number of iterations
TEST_ITERATIONS = 1
# Graph csv logger
CSVLOGGER = "training.log"


def get_csv_logger(tf_graphs_path):
    """
    This function writes csv logger
    :param tf_graphs_path: path directory for tf_graphs
    :return:
    """
    tf_graphs_path = os.path.join(tf_graphs_path, get_time_and_date())
    create_folder(tf_graphs_path)
    csv_file = open(os.path.join(tf_graphs_path, CSVLOGGER), "wb")
    csv_writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
    csv_writer.writeheader()
    return csv_writer, csv_file


def pre_process_data(entity, hierarchy_mapping_objects, objects_embeddings):
    """
    This function pre-processing the entities with the objects_embeddings to return RNN inputs and outputs
    :param entity: entity VG type
    :param hierarchy_mapping_objects: hierarchy_mapping of objects
    :param objects_embeddings: objects embedding - [150, 300]
    :return:
    """

    # Get the objects which the CNN has been chosen
    candidate_objects = np.argmax(entity.objects_probs, axis=1)
    # Create the objects as a one hot vector [num_objects, 150]
    objects_hot_vectors = np.eye(len(hierarchy_mapping_objects), dtype='uint8')[candidate_objects]
    # Get embedding per objects [num_objects, 150] * [150, 300] = [num_objects, 300]
    objects_embeddings = np.dot(objects_hot_vectors, objects_embeddings)
    # Get relationship embeddings
    rnn_inputs, rnn_outputs = get_rnn_data(entity, objects_embeddings)
    return rnn_inputs, rnn_outputs


def get_rnn_data(entity, objects_embeddings):
    """
    This function prepares the rnn inputs and outputs.
    :param entity: entity object
    :param objects_embeddings: objects embedding matrix: [num_objects, 300]
    :return: input: relationships embeddings - <object_i embeddings, object_i and object_j embeddings, object_j embeddings>
            output: relationships labels
    """
    relationships_embeddings_input = []
    relationships_embeddings_output = []
    for object_i_ind in range(len(entity.objects)):
        # Add object_i embeddings
        obj_i_embed = objects_embeddings[object_i_ind]
        for object_j_ind in range(len(entity.objects)):

            # if np.argmax(entity.predicates_labels[object_i_ind][object_j_ind]) == 50:
            #     continue

            # # Continue if it is the same object
            # if object_i_ind == object_j_ind:
            #     continue

            # Add object_j embeddings
            obj_j_embed = objects_embeddings[object_j_ind]

            # Add relation with concatenation with obj_i and obj_j
            obj_i_place = [entity.objects[object_i_ind].x, entity.objects[object_i_ind].y,
                           entity.objects[object_i_ind].width, entity.objects[object_i_ind].height]
            obj_j_place = [entity.objects[object_j_ind].x, entity.objects[object_j_ind].y,
                           entity.objects[object_j_ind].width, entity.objects[object_j_ind].height]
            objects_location = np.concatenate([obj_i_place, obj_j_place], axis=0)
            objects_location_with_padd = np.pad(objects_location, (0, NUM_INPUT -
                                                                   len(objects_location) % NUM_INPUT),
                                                'constant')
            # Adding RNN input [obj_i embedding, objects location with padding, obj_h embedding] -[3,300]
            rnn_input = np.vstack([obj_i_embed, objects_location_with_padd, obj_j_embed])
            relationships_embeddings_input.append(rnn_input)
            predicate_label_one_hot_vector = entity.predicates_labels[object_i_ind][object_j_ind]
            relationships_embeddings_output.append(predicate_label_one_hot_vector)

    relationships_embeddings_input = np.stack(relationships_embeddings_input)
    relationships_embeddings_output = np.stack(relationships_embeddings_output)
    return relationships_embeddings_input, relationships_embeddings_output


def train(name="test",
          nof_iterations=100,
          learning_rate=0.1,
          learning_rate_steps=1000,
          learning_rate_decay=0.5,
          load_module_name="module.ckpt",
          use_saved_module=False,
          timesteps=1,
          gpu=0,
          files_train_list=None,
          files_test_list=None):
    """

    :param files_test_list:
    :param files_train_list:
    :param name:
    :param nof_iterations:
    :param learning_rate:
    :param learning_rate_steps:
    :param learning_rate_decay:
    :param load_module_name:
    :param use_saved_module:
    :param timesteps:
    :param gpu:
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

    # Create Module
    language_module = LanguageModule(timesteps=timesteps, is_train=True, num_hidden=NUM_HIDDEN,
                                     num_classes=NOF_PREDICATES, num_input=NUM_INPUT, learning_rate=learning_rate,
                                     learning_rate_steps=learning_rate_steps, learning_rate_decay=learning_rate_decay)

    # get input place holders
    inputs_ph = language_module.get_inputs_placeholders()
    # get labels place holders
    labels_ph = language_module.get_labels_placeholders()
    # get learning rate place holder
    lr_ph = language_module.get_lr_placeholder()
    # get loss and train step
    loss, gradients, grad_placeholder, train_step = language_module.module_loss()
    # get module output
    accuracy = language_module.get_output()
    # get logits (likelihood)
    logits = language_module.get_logits()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Define Summaries
    tf_logs_path = FilesManager().get_file_path("language_module.train.tf_logs")
    summary_writer = tf.summary.FileWriter(tf_logs_path, graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()
    tf_graphs_path = FilesManager().get_file_path("language_module.train.tf_graphs")
    csv_writer, csv_file = get_csv_logger(tf_graphs_path)

    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = FilesManager().get_file_path("language_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index") and use_saved_module:
            saver.restore(sess, module_path_load)
            Logger().log("Model restored.")
        else:
            sess.run(init)

        # Get the entities
        entities_path = FilesManager().get_file_path("data.visual_genome.detections_v4")

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

        # Train module
        lr = learning_rate
        train_total_acc = 0
        best_test_loss = -1
        for epoch in range(1, nof_iterations):
            train_total_loss = 0
            train_acc = 0
            train_num_entities = 1
            steps = []
            # read data
            for file_dir in files_train_list:
                files = os.listdir(os.path.join(entities_path, file_dir))
                for file_name in files:
                    file_path = os.path.join(entities_path, file_dir, file_name)
                    file_handle = open(file_path, "rb")
                    train_entities = cPickle.load(file_handle)
                    file_handle.close()
                    for entity in train_entities:

                        if len(entity.relationships) == 0:
                            continue

                        # Pre-processing entities to get RNN inputs and outputs
                        rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects, objects_embeddings)

                        # Create the feed dictionary
                        feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs, lr_ph: lr}

                        # Run the network
                        accuracy_val, loss_val, gradients_val = sess.run([accuracy, loss, gradients], feed_dict=feed_dict)

                        # Append gradient to list (will be applied as a batch of entities)
                        steps.append(gradients_val)
                        # Calculates loss
                        train_total_loss += loss_val
                        # Calculates accuracy
                        train_acc += accuracy_val

                        # Update gradients in each epoch
                        if len(steps) == BATCH_SIZE:
                            for step in steps:
                                # apply steps
                                feed_grad_apply_dict = {grad_placeholder[j][0]: step[j][0] for j in
                                                        xrange(len(grad_placeholder))}
                                feed_grad_apply_dict[language_module.lr_ph] = lr
                                sess.run([train_step], feed_dict=feed_grad_apply_dict)
                            steps = []

                        # Update the number of entities
                        train_num_entities += 1

            # Update predicates accuracy
            train_total_acc += float(train_acc) / train_num_entities
            # Print stats
            logger.log("TRAIN: epoch: %d - loss: %f - predicates accuracy: %f - lr: %f" %
                       (epoch, train_total_loss, train_total_acc, lr))

            # Perform Testing
            if epoch % TEST_ITERATIONS == 0:
                # read data
                test_total_acc = 0
                test_total_loss = 0
                test_num_entities = 1

                for file_dir in files_test_list:
                    files = os.listdir(os.path.join(entities_path, file_dir))
                    for file_name in files:
                        file_path = os.path.join(entities_path, file_name)
                        file_handle = open(file_path, "rb")
                        test_entities = cPickle.load(file_handle)
                        file_handle.close()
                        for entity in test_entities:
                            # Pre-processing entities to get RNN inputs and outputs
                            rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects,
                                                                       objects_embeddings)

                            # Create the feed dictionary
                            feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs}

                            # Run the network
                            accuracy_val, loss_val = sess.run([accuracy, loss], feed_dict=feed_dict)

                            # Calculates loss
                            test_total_loss += loss_val
                            # Calculates accuracy
                            test_total_acc += accuracy_val
                            # Update the number of entities
                            test_num_entities += 1

                # Print stats
                Logger().log("TEST: iter: %d - loss: %f - predicates accuracy: %f " %
                             (epoch, test_total_loss, float(test_total_acc) / test_num_entities))
                # Write to CSV logger
                csv_writer.writerow({'epoch': epoch, 'acc': train_total_acc, 'loss': train_total_loss,
                                     'val_acc': float(test_total_acc) / test_num_entities, 'val_loss': test_total_loss})
                csv_file.flush()

                # save best module so far
                if best_test_loss == -1 or test_total_loss < best_test_loss:
                    module_path_save = os.path.join(module_path, name + "_best_module.ckpt")
                    save_path = saver.save(sess, module_path_save)
                    logger.log("Model saved in file: %s" % save_path)
                    best_test_loss = test_total_loss

            # Save module
            if epoch % SAVE_MODEL_ITERATIONS == 0:
                module_path_save = os.path.join(module_path, name + "_module.ckpt")
                save_path = saver.save(sess, module_path_save)
                logger.log("Model saved in file: %s" % save_path)

            # Update learning rate decay
            if epoch % learning_rate_steps == 0:
                lr *= learning_rate_decay

        # Save module
        module_path_save = os.path.join(module_path, name + "_module.ckpt")
        save_path = saver.save(sess, module_path_save)
        logger.log("Model saved in file: %s" % save_path)

        # Close csv logger
        csv_file.close()


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

        train(name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
              use_saved_model, timesteps, gpu, files_train_list, files_test_list)

        # p = Process(target=train, args=(
        #     name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
        #     use_saved_model, timesteps, gpu))
        # p.start()
        # processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
