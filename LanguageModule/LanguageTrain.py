import sys
sys.path.append("..")
import itertools
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
SAVE_MODEL_ITERATIONS = 10
# test every number of iterations
TEST_ITERATIONS = 1
# Graph csv logger
CSVLOGGER = "training.log"
# negative vs positive factor
POS_NEG_FACTOR = 3.3
POS_NEG_RATIO = 1
# Use Predcls task
PREDCLS = True


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


def pre_process_data(entity, hierarchy_mapping_objects, objects_embeddings, pos_neg_ratio=POS_NEG_RATIO, pred_cls=PREDCLS):
    """
    This function pre-processing the entities with the objects_embeddings to return RNN inputs and outputs
    :param pred_cls: using gt objects or predicted object (depends on PredCLS or SgCLS tasks)
    :param pos_neg_ratio:
    :param entity: entity VG type
    :param hierarchy_mapping_objects: hierarchy_mapping of objects
    :param objects_embeddings: objects embedding - [150, 300]
    :return:
    """

    # Check if we are using gt objects or predicted object (depends on PredCLS or SgCLS tasks)
    if not pred_cls:
        # SgCLS task - taking the predicted objects
        # Get the objects which the CNN has been chosen
        candidate_objects = np.argmax(entity.objects_probs, axis=1)
        # Create the objects as a one hot vector [num_objects, 150]
        objects_hot_vectors = np.eye(len(hierarchy_mapping_objects), dtype='uint8')[candidate_objects]
    else:
        # PredCLS task - taking the GT objects
        objects_hot_vectors = entity.objects_labels

    # Get embedding per objects [num_objects, 150] * [150, 300] = [num_objects, 300]
    objects_embeddings = np.dot(objects_hot_vectors, objects_embeddings)
    # Get relationship embeddings
    # rnn_inputs, rnn_outputs = get_rnn_data(entity, objects_embeddings, ignore_negatives=True)
    rnn_inputs, rnn_outputs = get_rnn_positive_data(entity, objects_embeddings, pos_neg_ratio)
    return rnn_inputs, rnn_outputs


def get_rnn_data(entity, objects_embeddings, ignore_negatives=False):
    """
    This function prepares the rnn inputs and outputs.
    :param ignore_negatives: Do we need to ignore negatives
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

            if ignore_negatives and np.argmax(entity.predicates_labels[object_i_ind][object_j_ind]) == NOF_PREDICATES - 1:
                continue

            # # Continue if it is the same object
            # if object_i_ind == object_j_ind:
            #     continue

            # Add object_j embeddings
            obj_j_embed = objects_embeddings[object_j_ind]

            # Take Predicates features from MASK CNN
            objects_location = entity.predicates_outputs_with_no_activation[object_i_ind][object_j_ind]

            # Add relation with concatenation with obj_i and obj_j
            # obj_i_place = [entity.objects[object_i_ind].x, entity.objects[object_i_ind].y,
            #                entity.objects[object_i_ind].width, entity.objects[object_i_ind].height]
            # obj_j_place = [entity.objects[object_j_ind].x, entity.objects[object_j_ind].y,
            #                entity.objects[object_j_ind].width, entity.objects[object_j_ind].height]
            # objects_location = np.concatenate([obj_i_place, obj_j_place], axis=0)
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


def get_rnn_positive_data(entity, objects_embeddings, pos_neg_ratio=POS_NEG_RATIO):
    """
    This function prepares the rnn inputs and outputs.
    :param pos_neg_ratio:
    :param entity: entity object
    :param objects_embeddings: objects embedding matrix: [num_objects, 300]
    :return: input: relationships embeddings - <object_i embeddings, object_i and object_j embeddings, object_j embeddings>
            output: relationships labels
    """
    relationships_embeddings_input = []
    relationships_embeddings_output = []
    rows, cols = np.where(entity.predicates_labels[:, :, NOF_PREDICATES - 1] != 1)

    # Number of positives
    num_pos = len(rows)
    # Add positives
    for ind in range(len(rows)):
        object_i_ind = rows[ind]
        object_j_ind = cols[ind]
        # Add object_i embeddings
        obj_i_embed = objects_embeddings[object_i_ind]
        # Add object_j embeddings
        obj_j_embed = objects_embeddings[object_j_ind]

        # Take Predicates features from MASK CNN
        objects_location = entity.predicates_outputs_with_no_activation[object_i_ind][object_j_ind]

        # Add relation with concatenation with obj_i and obj_j
        # obj_i_place = [entity.objects[object_i_ind].x, entity.objects[object_i_ind].y,
        #                entity.objects[object_i_ind].width, entity.objects[object_i_ind].height]
        # obj_j_place = [entity.objects[object_j_ind].x, entity.objects[object_j_ind].y,
        #                entity.objects[object_j_ind].width, entity.objects[object_j_ind].height]
        # objects_location = np.concatenate([obj_i_place, obj_j_place], axis=0)
        objects_location_with_padd = np.pad(objects_location, (0, NUM_INPUT -
                                                               len(objects_location) % NUM_INPUT),
                                            'constant')

        # Adding RNN input [obj_i embedding, objects location with padding, obj_h embedding] -[3,300]
        rnn_input = np.vstack([obj_i_embed, objects_location_with_padd, obj_j_embed])
        relationships_embeddings_input.append(rnn_input)
        predicate_label_one_hot_vector = entity.predicates_labels[object_i_ind][object_j_ind]
        relationships_embeddings_output.append(predicate_label_one_hot_vector)

    # Number of negatives
    num_neg = int(num_pos * pos_neg_ratio)

    # Set of total objects indices
    total_indices = set(range(len(entity.objects)))
    # Get the negative rows (total - positives)
    neg_rows_remains = total_indices - set(rows)
    # Get the negative cols (total - positives)
    neg_cols_remains = total_indices - set(cols)
    # Get the whole negative relations possible
    all_neg_relations = list(itertools.product(neg_rows_remains, neg_cols_remains))
    # Remove the same index
    all_neg_relations = [tup for tup in all_neg_relations if tup[0] != tup[1]]
    # Set the number of negatives which are want
    number = min(num_neg, len(all_neg_relations))
    # Shuffle
    np.random.shuffle(all_neg_relations)
    # Choose random negatives
    neg_tuples = all_neg_relations[:number]

    for object_i_ind, object_j_ind in neg_tuples:

        # Add object_i embeddings
        obj_i_embed = objects_embeddings[object_i_ind]
        # Add object_j embeddings
        obj_j_embed = objects_embeddings[object_j_ind]

        # Take Predicates features from MASK CNN
        objects_location = entity.predicates_outputs_with_no_activation[object_i_ind][object_j_ind]

        # Add relation with concatenation with obj_i and obj_j
        # obj_i_place = [entity.objects[object_i_ind].x, entity.objects[object_i_ind].y,
        #                entity.objects[object_i_ind].width, entity.objects[object_i_ind].height]
        # obj_j_place = [entity.objects[object_j_ind].x, entity.objects[object_j_ind].y,
        #                entity.objects[object_j_ind].width, entity.objects[object_j_ind].height]
        # objects_location = np.concatenate([obj_i_place, obj_j_place], axis=0)
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
    # Shuffle indices
    all_indice = range(len(relationships_embeddings_input))
    np.random.shuffle(all_indice)
    return relationships_embeddings_input[all_indice], relationships_embeddings_output[all_indice]


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

    Logger().log("    %s = %s" % ("PredCLS task", PREDCLS))
    Logger().log("    %s = %s" % ("POS_NEG_RATIO", POS_NEG_RATIO))

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

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=None)
    # Get timestamp
    timestamp = get_time_and_date()

    # Define Summaries
    tf_logs_path = FilesManager().get_file_path("language_module.train.tf_logs")
    summary_writer = tf.summary.FileWriter(tf_logs_path, graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()
    tf_graphs_path = FilesManager().get_file_path("language_module.train.tf_graphs")
    csv_writer, csv_file = get_csv_logger(tf_graphs_path, timestamp)

    Logger().log("Start Training")
    with tf.Session() as sess:
        # Restore variables from disk.
        module_path = FilesManager().get_file_path("language_module.train.saver")
        module_path_load = os.path.join(module_path, load_module_name)
        if os.path.exists(module_path_load + ".index") and use_saved_module:
            saver.restore(sess, module_path_load)
            Logger().log("Model restored.")
        else:
            create_folder(os.path.join(module_path, timestamp))
            sess.run(init)

        # Get the entities
        entities_path = FilesManager().get_file_path("data.visual_genome.detections_v4")
        # files_train_list = ["Sat_Nov_11_21:59:10_2017"]
        # files_test_list = ["Sat_Nov_11_21:59:10_2017"]

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
        lr = learning_rate
        best_test_loss = -1
        for epoch in range(1, nof_iterations):
            try:
                train_loss_epoch = 0
                train_loss_batch = 0
                train_acc_epoch = 0
                train_acc_batch = 0
                # train_num_predicates_epoch = 0
                # train_num_predicates_batch = 0
                train_num_entities = 0
                steps = []
                # region Training
                for file_dir in files_train_list:
                    files = os.listdir(os.path.join(entities_path, file_dir))
                    for file_name in files:

                        # Load only entities
                        if ".log" in file_name or "lang" in file_name:
                            continue

                        file_path = os.path.join(entities_path, file_dir, file_name)
                        file_handle = open(file_path, "rb")
                        train_entities = cPickle.load(file_handle)
                        file_handle.close()
                        for entity in train_entities:
                            try:

                                if len(entity.relationships) == 0:
                                    continue

                                # # Set diagonal to be neg in entity
                                # indices = set_diag_to_negatives(entity, predicate_neg)
                                # # Get coeff matrix
                                # coeff_factor = get_coeff_factor(entity, indices)
                                # coeff_factor_reshape = coeff_factor.reshape(-1)

                                # Pre-processing entities to get RNN inputs and outputs
                                rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects,
                                                                           objects_embeddings,
                                                                           pos_neg_ratio=POS_NEG_RATIO,
                                                                           pred_cls=PREDCLS)

                                # Create the feed dictionary
                                feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs, lr_ph: lr}
                                             # coeff_loss_ph: coeff_factor_reshape}

                                # Run the network
                                accuracy_val, loss_val, gradients_val = sess.run([accuracy, loss, gradients],
                                                                                 feed_dict=feed_dict)

                                # Append gradient to list (will be applied as a batch of entities)
                                steps.append(gradients_val)
                                # Calculates loss
                                train_loss_epoch += loss_val
                                train_loss_batch += loss_val
                                # Calculates accuracy
                                train_acc_epoch += accuracy_val
                                train_acc_batch += accuracy_val
                                # Append number of predicates
                                # train_num_predicates_epoch += rnn_outputs.shape[0]
                                # train_num_predicates_batch += rnn_outputs.shape[0]

                                # Update gradients in each epoch
                                if len(steps) == BATCH_SIZE:
                                    for step in steps:
                                        # apply steps
                                        feed_grad_apply_dict = {grad_placeholder[j][0]: step[j][0] for j in
                                                                xrange(len(grad_placeholder))}
                                        feed_grad_apply_dict[language_module.lr_ph] = lr
                                        sess.run([train_step], feed_dict=feed_grad_apply_dict)
                                    steps = []

                                    # Print stats
                                    logger.log("TRAIN MINI-BATCH: epoch: %d - batch : %d - loss: %f - "
                                               "predicates accuracy: %f" %
                                               (epoch, train_num_entities / BATCH_SIZE,
                                                float(train_loss_batch) / BATCH_SIZE,
                                                float(train_acc_batch) / BATCH_SIZE))
                                    train_acc_batch = 0
                                    train_loss_batch = 0
                                # Update the number of entities
                                train_num_entities += 1

                            except Exception as e:
                                logger.log(
                                        "Error: problem in Train. Epoch: {0}, image id: {1} Exception: {2}".format(epoch,
                                                                                                                  entity.image.id,
                                                                                                                  str(
                                                                                                                      e)))
                                continue

                # endregion
                # Finished training - one Epoch

                # Print Stats
                logger.log("TRAIN EPOCH: epoch: %d - loss: %f - predicates accuracy: %f - lr: %f" %
                           (epoch, float(train_loss_epoch) / train_num_entities,
                            float(train_acc_epoch) / train_num_entities, lr))

                # region Testing
                if epoch % TEST_ITERATIONS == 0:
                    # read data
                    test_total_acc = 0
                    test_total_loss = 0
                    test_num_entities = 0
                    # test_num_predicates_total = 0

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

                                    # # Set diagonal to be neg in entity
                                    # indices = set_diag_to_negatives(entity, predicate_neg)
                                    # # Get coeff matrix
                                    # coeff_factor = get_coeff_factor(entity, indices)
                                    # coeff_factor_reshape = coeff_factor.reshape(-1)

                                    # Pre-processing entities to get RNN inputs and outputs
                                    rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects,
                                                                               objects_embeddings,
                                                                               pos_neg_ratio=0)

                                    # Create the feed dictionary
                                    feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs}
                                                 # coeff_loss_ph: coeff_factor_reshape}

                                    # Run the network
                                    accuracy_val, loss_val = sess.run([accuracy, loss], feed_dict=feed_dict)

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
                    # Write to CSV logger
                    csv_writer.writerow({'epoch': epoch, 'acc': float(train_acc_epoch) / train_num_entities,
                                         'loss': float(train_loss_epoch) / train_num_entities,
                                         'val_acc': float(test_total_acc) / test_num_entities,
                                         'val_loss': float(test_total_loss) / test_num_entities})
                    csv_file.flush()

                    # save best module so far
                    if best_test_loss == -1 or test_total_loss < best_test_loss:
                        # Save the best module till 5 epoch as different name
                        if epoch < 5:
                            module_path_save = os.path.join(module_path, timestamp, name + "_best5_module.ckpt")
                            save_path = saver.save(sess, module_path_save)
                            logger.log("Model Best till 5 epoch saved in file: %s" % save_path)

                        module_path_save = os.path.join(module_path, timestamp, name + "_best_module.ckpt")
                        save_path = saver.save(sess, module_path_save)
                        logger.log("Model Best saved in file: %s" % save_path)
                        best_test_loss = test_total_loss

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
