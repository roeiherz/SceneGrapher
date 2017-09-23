import inspect
import os
import sys
from multiprocessing import Process

sys.path.append("..")

from FilesManager.FilesManager import FilesManager
from Module import Module
from Utils.Logger import Logger
import tensorflow as tf
import numpy as np
from FeaturesExtraction.Utils.data import get_filtered_data
from Train import NOF_OBJECTS, NOF_PREDICATES, VISUAL_FEATURES_OBJECT_SIZE, VISUAL_FEATURES_PREDICATE_SIZE, TEST_PERCENT
from Utils.ModuleDetection import ModuleDetection
import Scripts
import cPickle
from Train import predicate_class_recall

def eval_image(img, reverse_object_ids, reverse_predicate_ids, labels_predicate, labels_object, out_belief_predicate_val, out_belief_object_val, k=100):
    """
    Scene Graph Classification -
    R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet prediction in an image)
    :param labels_predicate: labels of image predicates (each one is one hot vector) - shape (N, N, NOF_PREDICATES)
    :param labels_object: labels of image objects (each one is one hot vector) - shape (N, NOF_OBJECTS)
    :param out_belief_predicate_val: belief of image predicates - shape (N, N, NOF_PREDICATES)
    :param out_belief_object_val: belief of image objects - shape (N, NOF_OBJECTS)
    :param k: k most confident predictions to consider
    :return: image score, number of the gt triplets that appear in the k most confident predictions,
                         number of the gt triplets
    """
    # create module detections
    detections = ModuleDetection(img, reverse_object_ids, reverse_predicate_ids)

    # iterate over each relation to predict and find k highest predictions
    top_predictions = np.zeros((0,))
    top_likelihoods = np.zeros((0,))
    top_k_global_subject_ids = np.zeros((0,))
    top_k_global_object_ids = np.zeros((0,))

    N = labels_object.shape[0]
    if N == 1:
        return 0, 0, 0

    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if (subject_index == object_index):
                continue

            predicate_prob = out_belief_predicate_val[subject_index][object_index]
            subject_prob = out_belief_object_val[subject_index]
            object_prob = out_belief_object_val[object_index]

            # calc tensor of probabilities of visual moudle
            predict_prob = np.multiply.outer(subject_prob, np.multiply.outer(predicate_prob.flatten(), object_prob))

            # remove negative probabilties
            predict_prob[:, NOF_PREDICATES - 1, :] = 0

            # get the highset probabilities
            #max_k_predictions = np.argsort(predict_prob.flatten())[-k:]
            max_k_predictions = np.argpartition(predict_prob.flatten(),-k)[-k:]
            max_k_predictions_triplets = np.unravel_index(max_k_predictions, predict_prob.shape)
            max_k_subjects = max_k_predictions_triplets[0]
            max_k_predicates = max_k_predictions_triplets[1]
            max_k_objects = max_k_predictions_triplets[2]
            max_k_likelihoods = predict_prob[max_k_subjects, max_k_predicates, max_k_objects]

            # append to the list of highest predictions
            top_predictions = np.concatenate((top_predictions, max_k_predictions))
            top_likelihoods = np.concatenate((top_likelihoods, max_k_likelihoods))

            # store the relevant subject and object
            max_k_global_subject_ids = np.ones(max_k_likelihoods.shape) * subject_index
            max_k_global_object_ids = np.ones(max_k_likelihoods.shape) * object_index
            top_k_global_subject_ids = np.concatenate((top_k_global_subject_ids, max_k_global_subject_ids))
            top_k_global_object_ids = np.concatenate((top_k_global_object_ids, max_k_global_object_ids))

    # get k highest confidence
    top_k_indices = np.argsort(top_likelihoods)[-k:]
    predictions = top_predictions[top_k_indices]
    global_sub_ids = top_k_global_subject_ids[top_k_indices]
    global_obj_ids = top_k_global_object_ids[top_k_indices]
    likelihoods = top_likelihoods[top_k_indices]
    triplets = np.unravel_index(predictions.astype(int), predict_prob.shape)
    #for i in range(k):
    #    detections.add_detection(global_subject_id=global_sub_ids[i], global_object_id=global_obj_ids[i],
    #                             pred_subject=triplets[0][i], pred_object=triplets[2][i], pred_predicate=triplets[1][i],
    #                             top_k_index=i, confidence=likelihoods[i])

    predicats_gt = np.argmax(labels_predicate, axis=2)
    objects_gt = np.argmax(labels_object, axis=1)

    img_score = 0
    nof_pos_relationship = 0
    for subject_index in range(N):
        for object_index in range(N):
            # filter if subject equals to object
            if (subject_index == object_index):
                continue
            # filter negative relationship
            if predicats_gt[subject_index, object_index] == NOF_PREDICATES - 1:
                continue

            nof_pos_relationship += 1
            predicate_id = predicats_gt[subject_index][object_index]
            sub_id = objects_gt[subject_index]
            obj_id = objects_gt[object_index]
            gt_relation = np.ravel_multi_index((sub_id, predicate_id, obj_id), predict_prob.shape)

            # filter the predictions for the specific subject
            sub_predictions_indices = set(np.where(global_sub_ids == subject_index)[0])
            obj_predictions_indices = set(np.where(global_obj_ids == object_index)[0])
            relation_indices = set(np.where(predictions == gt_relation)[0])

            indices = sub_predictions_indices & obj_predictions_indices & relation_indices
            if len(indices) != 0:
                img_score += 1
            else:
                img_score = img_score

    if nof_pos_relationship != 0:
        img_score_precent = float(img_score) / nof_pos_relationship
    else:
        img_score_precent = 0

    #detections.save_stat(score=img_score_precent)

    return img_score_precent, img_score, nof_pos_relationship


def eval(load_module_name=None, k_recall=True, pred_class=True, rnn_steps=0, k=100, gpu=1):
    """
    Evaluate module:
    - Scene Graph Classification - R@k metric (measures the fraction of ground truth relationships
      triplets that appear among the k most confident triplet predirction in an image)
    - Predicate Classification - Examine the model performance on predicates classification in isolation from other factors
    :param nof_iterations: - nof of images to test
    :param load_module_name: name of the module to load
    :param gpu: gpu number to use
    :return: nothing - output to logger instead
    """
    filesmanager = FilesManager()
    # create logger
    logger = Logger()

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
                    visual_features_object_size=VISUAL_FEATURES_OBJECT_SIZE, is_train=False,
                    rnn_steps=rnn_steps)

    # get input place holders
    belief_predicate_ph, belief_object_ph, extended_belief_object_shape_ph, visual_features_predicate_ph, visual_features_object_ph = module.get_in_ph()
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

    # read data
    entities_path = filesmanager.get_file_path("data.visual_genome.detections_v4")
    files_list = ["Thu_Aug_24_13:07:21_2017/predicated_entities_0_to_1000.p"] #,"Wed_Aug__9_10:04:43_2017/predicated_entities_0_to_1000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_1000_to_2000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_2000_to_3000.p", "Wed_Aug__9_10:04:43_2017/predicated_entities_3000_to_4000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_0_to_1000.p", "Tue_Aug__8_23:28:18_2017/predicated_entities_1000_to_2000.p"]
    img_ids = Scripts.get_img_ids()
    _, object_ids, predicate_ids = get_filtered_data(filtered_data_file_name="mini_filtered_data", category='entities_visual_module')
    reverse_object_ids = {object_ids[id]: id for id in object_ids}
    reverse_predicate_ids = {predicate_ids[id]: id for id in predicate_ids}

    with tf.Session() as sess:
        if load_module_name != None:
            # Restore variables from disk.
            module_path = filesmanager.get_file_path("sg_module.train.saver")
            module_path_load = os.path.join(module_path, load_module_name + "_module.ckpt")
            if os.path.exists(module_path_load + ".index"):
                saver.restore(sess, module_path_load)
                logger.log("Model restored.")
            else:
                raise Exception("Module not found")
        else:
            sess.run(init)
        # eval module
        correct_predicate = np.zeros(NOF_PREDICATES)
        total_predicate = np.zeros(NOF_PREDICATES)
        correct = 0
        total = 0
        # create one hot vector for predicate_neg
        predicate_neg = np.zeros(NOF_PREDICATES)
        predicate_neg[NOF_PREDICATES - 1] = 1
        index = 0
        for file_name in files_list:
            file_path = os.path.join(entities_path, file_name)
            file_handle = open(file_path, "rb`")
            test_entities = cPickle.load(file_handle)
            file_handle.close()
            for entity in test_entities:

                # set diagonal to be neg
                indices = np.arange(entity.predicates_probes.shape[0])
                entity.predicates_outputs_with_no_activation[indices, indices, :] = predicate_neg
                entity.predicates_labels[indices, indices, :] = predicate_neg

                # get shape of extended object to be used by the module
                extended_belief_object_shape = np.asarray(entity.predicates_probes.shape)
                extended_belief_object_shape[2] = NOF_OBJECTS
                
                # use object class labels for pred class (multiply be some factor to convert to belief)                
                if pred_class:
                    in_object_belief = entity.objects_labels * 10
                else:
                    in_object_belief = entity.objects_outputs_with_no_activations

                # create the feed dictionary
                feed_dict = {belief_predicate_ph: entity.predicates_outputs_with_no_activation, belief_object_ph: in_object_belief,
                             extended_belief_object_shape_ph: extended_belief_object_shape,
                             visual_features_predicate_ph: entity.predicates_features,
                             visual_features_object_ph: entity.objects_features}

                out_predicate_probes_val, out_object_probes_val = \
                    sess.run([out_predicate_probes, out_object_probes],
                             feed_dict=feed_dict)
                # set diag in order to take in statistic  
                out_predicate_probes_val[indices, indices, :] = predicate_neg
                
                # use object class labels for pred class                
                if pred_class:
                    out_object_probes_val = entity.objects_labels
                
                # eval image
                if k_recall:
                    k_metric_res, correct_image, total_image = eval_image(entity, reverse_object_ids, reverse_predicate_ids, entity.predicates_labels, entity.objects_labels, out_predicate_probes_val, out_object_probes_val, k=k)
                    correct += correct_image
                    total += total_image
                    total_score = float(correct) / total
                    logger.log("result %d - %f (%d / %d) - total %f" % (index, k_metric_res, correct_image, total_image, total_score))

                # eval per predicate
                correct_predicate_image, total_predicate_image = predicate_class_recall(entity.predicates_labels,
                                                                                        out_predicate_probes_val)
                correct_predicate += correct_predicate_image
                total_predicate += total_predicate_image
                index += 1


        for i in range(NOF_PREDICATES):
            if total_predicate[i] != 0:
                logger.log("{0} recall@5 is {1} (total - {2}, correct {3})".format(reverse_predicate_ids[i],
                                                                                   float(correct_predicate[i]) /
                                                                                   total_predicate[i],
                                                                                   total_predicate[i],
                                                                                   correct_predicate[i]))

        print("Final Result for pred_class=%s k=%d - %f" % (str(pred_class), k, total_score))

    print("Debug")


if __name__ == "__main__":
    load_module_name="obj_test_new2_best"
    k_recall=True
    rnn_steps=2
    gpu=2
     
    
    pred_class=False
    k=100
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    exit()
    
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    
    exit()
    pred_class=True
    k=50
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()
    

    pred_class=True
    k=100
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()

    pred_class=True
    k=50
    p = Process(target=eval, args=(load_module_name, k_recall, pred_class, rnn_steps, k, gpu))
    p.start()
    p.join()

