import sys

sys.path.append("..")
import pandas as pd
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
    rnn_inputs, rnn_outputs = get_rnn_full_data(entity, objects_embeddings)
    return rnn_inputs, rnn_outputs


def get_rnn_full_data(entity, objects_embeddings):
    """
    This function prepares the rnn inputs and outputs for the whole data for prediction
    :param entity: entity object
    :param objects_embeddings: objects embedding matrix: [num_objects, 300]
    :return: input: relationships embeddings - <object_i embeddings, object_i and object_j embeddings, object_j embeddings>
            output: relationships labels
    """
    relationships_embeddings_input = []
    relationships_embeddings_output = []
    num_objects = len(entity.objects)

    for object_i_ind in range(num_objects):
        # Add object_i embeddings
        obj_i_embed = objects_embeddings[object_i_ind]
        for object_j_ind in range(num_objects):
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


def predict(nof_iterations=100,
            learning_rate=0.1,
            learning_rate_steps=1000,
            learning_rate_decay=0.5,
            load_module_name="module.ckpt",
            timesteps=1,
            gpu=0,
            files_train_list=None,
            files_test_list=None,
            print_stats=False,
            save_pickles=False):
    """

    :param save_pickles: To save new pickles for each file
    :param print_stats: print stats for logger
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

    # Create Module
    language_module = LanguageModule(timesteps=timesteps, is_train=True, num_hidden=NUM_HIDDEN,
                                     num_classes=NOF_PREDICATES, num_input=NUM_INPUT, learning_rate=learning_rate,
                                     learning_rate_steps=learning_rate_steps, learning_rate_decay=learning_rate_decay)

    # get input place holders
    inputs_ph = language_module.get_inputs_placeholders()
    # get labels place holders
    labels_ph = language_module.get_labels_placeholders()
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
        files_list = files_test_list + files_train_list
        files_list = ["Sat_Nov_11_21:59:10_2017"]

        if files_list is None or len(files_list) == 0:
            Logger().log("Error: No data for prediction")
            return None

        # Load hierarchy_mappings
        hierarchy_mapping_objects = FilesManager().load_file("data.visual_genome.hierarchy_mapping_objects")
        hierarchy_mapping_predicates = FilesManager().load_file("data.visual_genome.hierarchy_mapping_predicates")
        # Load pre-trained objects embeddings
        objects_embeddings = FilesManager().load_file("language_module.word2vec.object_embeddings")
        # Load img_id_to_split
        img_id_to_split = FilesManager().load_file("data.visual_genome.img_id_to_split")

        # Create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1

        test_total_acc = 0
        test_total_loss = 0
        test_num_entities = 0
        iter = 0
        # test_num_predicates_total = 0

        # Dict of stats per image id
        stats_dict_new = {}
        stats_dict_old = {}
        # Get data frame
        df = get_data_frame()

        for file_dir in files_list:
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
                        iter += 1
                        logger.log('Predicting image id {0} in iteration {1} \n'.format(entity.image.id, iter))

                        if len(entity.relationships) == 0:
                            continue

                        # Pre-processing entities to get RNN inputs and outputs
                        rnn_inputs, rnn_outputs = pre_process_data(entity, hierarchy_mapping_objects,
                                                                   objects_embeddings)

                        # Create the feed dictionary
                        feed_dict = {inputs_ph: rnn_inputs, labels_ph: rnn_outputs}

                        # Run the network
                        accuracy_val, loss_val, logits_val, softmax_predictions_val = sess.run(
                            [accuracy, loss, logits, softmax_predictions],
                            feed_dict=feed_dict)

                        # Get new and old predictions and gt labels
                        # soft_max = softmax(logits_val)
                        predictions = np.argmax(softmax_predictions_val, axis=1)
                        previous_predictions = np.argmax(entity.predicates_probes, axis=2).reshape(-1)
                        gt_labels = np.argmax(entity.predicates_labels, axis=2).reshape(-1)

                        # Print and Calc stats for new predictions
                        df = print_and_calc_stats(stats_dict_new, df, accuracy_val, gt_labels, predictions,
                                                  entity.image.id, img_id_to_split[entity.image.id],
                                                  print_stats, old_predictions=False)
                        # Print and Calc stats for old predictions
                        df = print_and_calc_stats(stats_dict_old, df,
                                                  np.sum(previous_predictions == gt_labels) / float(len(gt_labels)),
                                                  gt_labels, previous_predictions, entity.image.id,
                                                  img_id_to_split[entity.image.id], print_stats,
                                                  old_predictions=True)

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
                            "Exception in iter: {0}, image id: {1} Exception: {2}".format(iter, entity.image.id,
                                                                                          str(e)))
                        continue

        # Print total stats
        Logger().log("TEST EPOCH: loss: %f - predicates accuracy: %f " %
                     (float(test_total_loss) / test_num_entities,
                      float(test_total_acc) / test_num_entities))

        # Save data dicts
        module_path_save = os.path.join(module_path, load_module_name.split("/")[0])

        # stats_dict_new
        fl = open(os.path.join(module_path_save, "predict_stats_dict_new.p"), "wb")
        cPickle.dump(stats_dict_new, fl)
        fl.close()
        # stats_dict_old
        fl = open(os.path.join(module_path_save, "predict_stats_dict_old.p"), "wb")
        cPickle.dump(stats_dict_old, fl)
        fl.close()

        # Save DataFrame and csv
        df.to_csv(os.path.join(module_path_save, "predict.csv"))
        fl = open(os.path.join(module_path_save, "predict_df.p"), "wb")
        cPickle.dump(df, fl)
        fl.close()

        # Finished Testing


def get_data_frame():
    """
    This function will create data frame foreach entity
    :return: data frame
    """

    # Define the rows for the DataFrame
    dataframe_labels = ["Image_Id", "Total_Relations", "Number_Of_Positive_Relations",
                        "Number_Of_Negative_Relations", "Relations_Accuracy", "Positive_Relations_Accuracy",
                        "Negative_Relations_Accuracy", "Old_Predictions", "Train_Entity"]

    # Define DataFrame
    df = pd.DataFrame(columns=dataframe_labels)
    return df


def print_and_calc_stats(stats_dict, df, accuracy_val, gt_labels, predictions, image_id, img_id_split, print_stats,
                         old_predictions=False):
    """
    This function calc stats while prediction relations
    :param print_stats: a flag for print stats or not
    :param img_id_split: 0 for train 2 test entity
    :param df: data frame to add the results
    :param old_predictions: a flag to set old prediction or new prediction
    :param stats_dict: dict of stats
    :param image_id: entity image id
    :param accuracy_val: accuracy from the BI-RNN
    :param gt_labels: the gt labels of the relations
    :param predictions: the predictions of the relations
    :return data frame
    """

    total_num_relations = len(gt_labels)
    num_negative_relations = np.sum(gt_labels == NOF_PREDICATES - 1)
    num_positive_relations = np.sum(gt_labels != NOF_PREDICATES - 1)
    positive_indices = np.where(gt_labels != NOF_PREDICATES - 1)
    negative_indices = np.where(gt_labels == NOF_PREDICATES - 1)
    positive_predictions = predictions[positive_indices]
    positive_gt_labels = gt_labels[positive_indices]
    negative_predictions = predictions[negative_indices]
    negative_gt_labels = gt_labels[negative_indices]
    positive_acc_relations = np.sum(positive_predictions == positive_gt_labels) / float(num_positive_relations)
    negative_acc_relations = np.sum(negative_predictions == negative_gt_labels) / float(num_negative_relations)
    # img_id to split is 0 for train 2 for test
    train_entity = True if img_id_split == 0 else False

    # Print stats
    if print_stats:
        if old_predictions:
            logger.log("Old predictions:")
        else:
            logger.log("New predictions:")

        logger.log("The Total number of Relations is {0} while {1} of them positives and {2} "
                   "of them negatives ".format(total_num_relations, num_positive_relations,
                                               num_negative_relations))
        logger.log("The Total Relations accuracy is {0}".format(accuracy_val))
        logger.log("The Positive Relations accuracy is {0}".format(positive_acc_relations))
        logger.log("The Negative Relations accuracy is {0}".format(negative_acc_relations))

    # Calc stats
    stats_dict[image_id] = {"Total_Relations": total_num_relations,
                            "Number_Of_Positive_Relations": num_positive_relations,
                            "Number_Of_Negative_Relations": num_negative_relations, "Relations_Accuracy": accuracy_val,
                            "Positive_Relations_Accuracy": positive_acc_relations,
                            "Negative_Relations_Accuracy": negative_acc_relations, "Train_Entity": train_entity}

    row_data = {"Image_Id": image_id, "Total_Relations": total_num_relations,
                "Number_Of_Positive_Relations": num_positive_relations,
                "Number_Of_Negative_Relations": num_negative_relations, "Relations_Accuracy": accuracy_val,
                "Positive_Relations_Accuracy": positive_acc_relations,
                "Negative_Relations_Accuracy": negative_acc_relations, "Old_Predictions": old_predictions,
                "Train_Entity": train_entity}

    # Adding a row to the data frame
    df.loc[-1] = row_data
    # Shifting index
    df.index = df.index + 1
    # Sorting by index
    df = df.sort()
    return df


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
        print_stats = process_params["print_stats"]
        save_pickles = process_params["save_pickles"]

        predict(nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name, timesteps,
                gpu, files_train_list, files_test_list, print_stats, save_pickles)

        # p = Process(target=train, args=(
        #     name, nof_iterations, learning_rate, learning_rate_steps, learning_rate_decay, load_model_name,
        #     use_saved_model, timesteps, gpu))
        # p.start()
        # processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()
