from __future__ import print_function

from DesignPatterns.Detections import Detections
from FeaturesExtraction.Lib.VisualGenomeDataGenerator import visual_genome_data_parallel_generator_with_batch, \
    visual_genome_data_predicate_generator_with_batch
from FeaturesExtraction.Lib.Zoo import ModelZoo
import traceback
import os
import cPickle
import numpy as np
from FeaturesExtraction.Lib.Config import Config
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
import sys
import math
from FeaturesExtraction.Utils.Boxes import BOX
from FeaturesExtraction.Utils.Utils import VG_VisualModule_PICKLES_PATH, get_img_resize, TRAINING_OBJECTS_CNN_PATH, \
    TRAINING_PREDICATE_CNN_PATH, WEIGHTS_NAME, get_img
import time
from FeaturesExtraction.Utils.data import get_filtered_data, get_name_from_file, process_to_detections
from FeaturesExtraction.Utils.Utils import DATA, VISUAL_GENOME
from FilesManager.FilesManager import FilesManager
from TrainPredicateCNN import preprocessing_relations, pick_different_negative_sample_ratio
from Utils.Logger import Logger

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2
NUM_EPOCHS = 1
NUM_BATCHES = 128
RATIO = 3.0 / 10

# If the allocation of training, validation and testing does not adds up to one
used_percent = TRAINING_PERCENT + VALIDATION_PERCENT + TESTING_PERCENT
if not used_percent == 1:
    error_msg = 'Data used percent (train + test + validation) is {0} and should be 1'.format(used_percent)
    print(error_msg)
    raise Exception(error_msg)

__author__ = 'roeih'


def get_resize_images_array(detections, config):
    """
    This function calculates the resize image for each detection and returns a numpy ndarray
    :param detections: a numpy Detections dtype array
    :return: a numpy array of shape (len(detections), config.crop_width, config.crop_height , 3)
    """
    resized_img_lst = []
    ind = 1

    for detection in detections:
        try:
            box = detection[Detections.UnionBox]
            url_data = detection[Detections.Url]
            img = get_img(url_data)
            patch = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
            resized_img = get_img_resize(patch, config.crop_width, config.crop_height, type=config.padding_method)
            resized_img_lst[detection[Detections.Id]] = resized_img

            if ind % 10000 == 0:
                logger.log("Proccessed 1000 detections to union box")

            ind += 1

        except Exception as e:
            logger.log("Exception for detection_id: {0}, image: {1}".format(detection[Detections.Id],
                                                                            detection[Detections.Url]))
            logger.log(str(e))
            traceback.print_exc()
            resized_img_lst.append(np.zeros((config.crop_width, config.crop_height, 3)))
    return np.array(resized_img_lst)


def load_full_detections(detections_file_name):
    """
    This function gets the whole filtered detections data (with no split between the  modules)
    :return: detections
    """
    # Check if pickles are already created
    detections_path = FilesManager().get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(detections_file_name)))

    if os.path.isfile(detections_path):
        logger.log('Detections numpy array is Loading from: {0}'.format(detections_path))
        detections = cPickle.load(open(detections_path, 'rb'))
        return detections

    return None


def get_model(number_of_classes, weight_path, config):
    """
        This function loads the model
        :param weight_path: model weights path
        :param number_of_classes: number of classes
        :param config: config file
        :return: model
        """

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (config.crop_height, config.crop_width, 3)

    img_input = Input(shape=input_shape_img, name="image_input")

    # Define ResNet50 model Without Top
    net = ModelZoo()
    model_resnet50 = net.resnet50_base(img_input, trainable=True)
    model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
    output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
        model_resnet50)

    # Define the model
    model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
    # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    model.summary()

    # Load pre-trained weights for ResNet50
    try:
        logger.log("Start loading Weights")
        model.load_weights(weight_path, by_name=True)
        logger.log('Finished successfully loading weights from {}'.format(weight_path))

    except Exception as e:
        logger.log('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        raise Exception(e)

    logger.log('Finished successfully loading Model')
    return model


def save_files(files, name=""):
    """
    This function save the files 
    """

    # Save detections
    detections_filename = open(name, 'wb')
    # Pickle detections
    cPickle.dump(files, detections_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    detections_filename.close()
    logger.log("File Have been save in {}".format(detections_filename))


def sort_detections_by_url(detections):
    """
    This function removes detections with specific indices
    :param detections: detections numpy dtype array
    :return: sorted detections 
    """
    idx = np.where((detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K/2321818.jpg") |
                   (detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K/2334844.jpg") |
                   (detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K_2/3807.jpg") |
                   (detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K_2/2410658.jpg") |
                   (detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K/2374264.jpg"))
    new_detections = np.delete(detections, idx)
    return new_detections


def load_predicts(file_name=""):
    """
    This function load the detection after predicated predicates and check for some statistics
    :return: 
    """

    # load file
    detections_file = open(os.path.join(VG_VisualModule_PICKLES_PATH, file_name))
    # Pickle detections
    detections = cPickle.load(detections_file)

    tt = np.where((detections[Detections.PredictSubjectClassifications] == detections[Detections.Predicate]) &
                  (detections[Detections.Predicate] != u'neg'))
    logger.log('debug')


def save_weights(predict_model, file_name=""):
    """
    This function save the weights of the last layer of Predicate Networks for late initialization in full training model
    :param predict_model: the predict model
    :param file_name: file name string which will be saved
    :return: 
    """
    weights = predict_model.layers[-1].get_weights()[0]  # [2048, nof_classes=51]
    save_files(weights, file_name)
    logger.log("Saved the last layer weights [2048,51] ")


if __name__ == '__main__':

    # Define FileManager
    filemanager = FilesManager()
    # Define Logger
    logger = Logger()

    # Get argument
    if len(sys.argv) < 4:
        # Default GPU number
        gpu_num = 0
        objects_training_dir_name = ""
        predicates_training_dir_name = ""
    else:
        # Get the GPU number from the user
        gpu_num = sys.argv[1]
        objects_training_dir_name = sys.argv[2]
        logger.log("Object training folder parameter is: {}".format(objects_training_dir_name))
        predicates_training_dir_name = sys.argv[3]
        logger.log("Predicate training folder parameter is: {}".format(predicates_training_dir_name))

    # Load class config
    config = Config(gpu_num)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)

    # BOTH MODULE + VISUAL
    # detections = load_full_detections(detections_file_name="mini_detections")
    # detections = sort_detections_by_url(detections)

    # Load detections dtype numpy array and hierarchy mappings
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "mini_filtered_data",
                                                                                          category='entities_module')

    # Get Visual Genome Data relations
    relations = preprocessing_relations(entities, hierarchy_mapping_objects, hierarchy_mapping_predicates,
                                        relation_file_name="mini_relations_module")

    # Process relations to numpy Detections dtype
    detections = process_to_detections(relations, detections_file_name="mini_detections_module")

    detections = sort_detections_by_url(detections)

    # Check the training folders from which we take the weights aren't empty
    if not objects_training_dir_name or not predicates_training_dir_name:
        logger.log("Error: No object training folder or predicate training folder has been given")
        exit()

    # Load the weight paths
    objects_model_weight_path = os.path.join(TRAINING_OBJECTS_CNN_PATH, objects_training_dir_name,
                                             WEIGHTS_NAME)
    predicates_model_weight_path = os.path.join(TRAINING_PREDICATE_CNN_PATH, predicates_training_dir_name,
                                                WEIGHTS_NAME)

    # Set the number of classes
    number_of_classes_objects = len(hierarchy_mapping_objects)

    if config.only_pos and "neg" in hierarchy_mapping_predicates:
        # Remove negative label from hierarchy_mapping_predicates because we want to train only positive
        hierarchy_mapping_predicates.pop("neg")
        detections = detections[detections[Detections.Predicate] != "neg"]
        RATIO = 0.0

    # Get new negative - positive ratio
    detections = pick_different_negative_sample_ratio(detections, ratio=RATIO)

    logger.log('Number of detections after sorting negatives: {0} with RATIO: {1}'.format(len(detections), RATIO))

    number_of_classes_predicates = len(hierarchy_mapping_predicates)

    # Create a data generator for VisualGenome for OBJECTS
    data_gen_val_objects_vg = visual_genome_data_parallel_generator_with_batch(data=detections,
                                                                               hierarchy_mapping=hierarchy_mapping_objects,
                                                                               config=config, mode='valid',
                                                                               batch_size=NUM_BATCHES)

    # Create a data generator for VisualGenome for PREDICATES
    data_gen_val_predicates_vg = visual_genome_data_predicate_generator_with_batch(data=detections,
                                                                                   hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                   config=config, mode='valid',
                                                                                   classification=Detections.Predicate,
                                                                                   type_box=Detections.UnionBox,
                                                                                   batch_size=NUM_BATCHES,
                                                                                   evaluate=True)

    # Get the object and predicate model
    object_model = get_model(number_of_classes_objects, weight_path=objects_model_weight_path, config=config)
    predicate_model = get_model(number_of_classes_predicates, weight_path=predicates_model_weight_path, config=config)

    # Save Weights
    # save_weights(predict_model, file_name="last_layer_ratio3_weights.p")

    logger.log('Starting Prediction')

    # region
    # Load Predicates
    # load_predicts(file_name="mini_predicated_predicates_with_neg_ratio1_Wed_Jun_14_20:25:16_2017.p")

    # Predict Predicates for some statistics
    logger.log('Predicting Probabilities - Predicates')

    predicted_objects = predicate_model.predict_generator(data_gen_val_predicates_vg,
                                                          steps=int(math.ceil(len(detections) / float(NUM_BATCHES))),
                                                          max_q_size=1, workers=1)
    logger.log("Saving Predicates Probabilities")
    save_files(predicted_objects, name="mini_predicated_predicates_010817.p")
    logger.log("Finished successfully saving Predicates Probabilities")
    # Get the max argument
    index_predicates_labels_per_sample = np.argmax(predicted_objects, axis=1)
    # Get the inverse-mapping: int id to str label
    index_to_label_mapping_predicates = {label: id for id, label in hierarchy_mapping_predicates.iteritems()}
    labels_per_sample = np.array(
        [index_to_label_mapping_predicates[label] for label in index_predicates_labels_per_sample])
    # Save detections in PredictSubjectClassifications
    detections[Detections.PredictSubjectClassifications] = labels_per_sample[:len(detections)]
    # Save detections
    logger.log("Saving predicates detections")
    save_files(detections, name="mini_predicated_predicates_010817.p")
    logger.log("Finished successfully saving predicated_detections")

    exit()
    # endregion

    logger.log('Predicting Probabilities - Objects')
    # Probabilities: [nof_detections * 2, 150]
    objects_probes = object_model.predict_generator(data_gen_val_objects_vg,
                                                    steps=int(math.ceil(len(detections) / float(NUM_BATCHES))),
                                                    max_q_size=1, workers=1)
    logger.log("Saving Objects Probabilities")
    objects_probes_path = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "mini_objects_with_probs"))
    filemanager.save_file(objects_probes_path, objects_probes)
    logger.log("Finished successfully saving Objects Probabilities")

    # Slice the Subject prob (even index)
    detections[Detections.SubjectConfidence] = np.split(objects_probes[::2], len(detections), axis=0)
    # Slice the Object prob (odd index)
    detections[Detections.ObjectConfidence] = np.split(objects_probes[1::2], len(detections), axis=0)
    # Get the max probes for each sample
    probes_per_sample = np.max(objects_probes, axis=1)
    # Get the max argument
    index_labels_per_sample = np.argmax(objects_probes, axis=1)

    # Get the inverse-mapping: int id to str label
    index_to_label_mapping = {label: id for id, label in hierarchy_mapping_objects.iteritems()}
    labels_per_sample = np.array([index_to_label_mapping[label] for label in index_labels_per_sample])

    # Slice the predicated Subject id (even index)
    detections[Detections.PredictSubjectClassifications] = labels_per_sample[::2]
    # Slice the predicated Object id (odd index)
    detections[Detections.PredictObjectClassifications] = labels_per_sample[1::2]
    logger.log('Finished Predicting Probabilities Successfully')

    # Save detections
    logger.log("Saving predicated_detections")
    detections_probes_path = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "mini_detections_with_probs"))
    filemanager.save_file(detections_probes_path, detections)
    logger.log("Finished successfully saving predicated_detections")

    # Get the Union-Box Features
    # resized_img_mat = get_resize_images_array(detections, config)

    logger.log('Calculating Union-Box Features')
    # Define the function
    get_features_output_func = K.function([predicate_model.layers[0].input], [predicate_model.layers[-2].output])
    ind = 0
    # Start measure time
    start = time.time()

    for detection in detections:
        try:

            ind += 1
            box = detection[Detections.UnionBox]
            url_data = detection[Detections.Url]
            img = get_img(url_data)
            patch = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
            resized_img = get_img_resize(patch, config.crop_width, config.crop_height, type=config.padding_method)
            resized_img = np.expand_dims(resized_img, axis=0)
            features_model = get_features_output_func([resized_img])[0]
            detection[Detections.UnionFeature] = features_model
            # detection[Detections.UnionFeature] = np.split(features_model, len(detections), axis=0)

            if ind % 10000 == 0:
                logger.log("Iteration Number: {}".format(ind))
                end = time.time()
                logger.log("Proccessed 10000 detections to union features in time {}s ".format(end - start))
                start = end

        except Exception as e:
            logger.log("Exception for detection_id: {0}, image: {1}".format(detection[Detections.Id],
                                                                            detection[Detections.Url]))
            logger.log(str(e))
            traceback.print_exc()

    logger.log("Finished to predict probabilities and union features")

    # Save detections
    logger.log("Saving predicated_detections")
    filemanager.save_file(detections_probes_path, detections)
    logger.log("Finished successfully saving predicated_detections")
