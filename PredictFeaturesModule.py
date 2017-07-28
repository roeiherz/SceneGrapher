from __future__ import print_function

from Data.VisualGenome.models import ObjectMapping
from DesignPatterns.Detections import Detections
from FeaturesExtraction.Lib.VisualGenomeDataGenerator import visual_genome_data_cnn_generator_with_batch, \
    visual_genome_data_predicate_pairs_generator_with_batch

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
from FeaturesExtraction.Utils.Boxes import BOX, find_union_box
from FeaturesExtraction.Utils.Utils import VG_VisualModule_PICKLES_PATH, get_img_resize, TRAINING_OBJECTS_CNN_PATH, \
    TRAINING_PREDICATE_CNN_PATH, WEIGHTS_NAME, get_img, get_mask_from_object, get_time_and_date, PREDICATED_FEATURES_PATH
from FeaturesExtraction.Utils.data import get_filtered_data, get_name_from_file
from FeaturesExtraction.Utils.Utils import DATA, VISUAL_GENOME
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger
import itertools
from Utils.Utils import create_folder

NUM_EPOCHS = 1
NUM_BATCHES = 128

__author__ = 'roeih'


def load_full_detections(detections_file_name):
    """
    This function gets the whole filtered detections data (with no split between the  modules)
    :return: detections
    """
    # Check if pickles are already created
    detections_path = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(detections_file_name)))

    if os.path.isfile(detections_path):
        logger.log('Detections numpy array is Loading from: {0}'.format(detections_path))
        detections = cPickle.load(open(detections_path, 'rb'))
        return detections

    return None


def save_files(path, files):
    """
    This function save the files
    """

    # Save detections
    detections_filename = open(os.path.join(path, "predicated_entities.p"), 'wb')
    # Pickle detections
    cPickle.dump(files, detections_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    detections_filename.close()
    logger.log("File Have been save in {}".format(detections_filename))


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


def from_object_to_objects_mapping(objects, correct_labels, url):
    """
    This function get objects from entities and transforms it to object mapping list
    :param objects: the objects from entities (per image)
    :param correct_labels: the hierarchy mapping
    :param url: the url field
    :return: objects_lst
    """

    # Initialized new objects list
    objects_lst = []
    for object in objects:

        # Get the label of object
        label = object.names[0]

        # Check if it is a correct label
        if label not in correct_labels:
            continue

        new_object_mapping = ObjectMapping(object.id, object.x, object.y, object.width, object.height, object.names,
                                           object.synsets, url)
        # Append the new objectMapping to objects_lst
        objects_lst.append(new_object_mapping)

    return objects_lst


def predict_objects_for_module(entity, objects, url_data, hierarchy_mapping_objects):
    """
    This function predicts objects for module later - object_probes [n, 150], object_features [n, 2048], object_labels
    [n, 150]
    :param objects: List of objects
    :param entity: Entity visual genome class
    :param url_data: a url data
    :param hierarchy_mapping_objects: hierarchy_mapping for objects
    :return: 
    """

    ## Get probabilities
    # Create a data generator for VisualGenome for OBJECTS
    data_gen_val_objects_vg = visual_genome_data_cnn_generator_with_batch(data=objects,
                                                                          hierarchy_mapping=hierarchy_mapping_objects,
                                                                          config=config, mode='validation',
                                                                          batch_size=NUM_BATCHES, evaluate=True)
    # Get the object probabilities [len(objects), 150]
    objects_probes = object_model.predict_generator(data_gen_val_objects_vg,
                                                    steps=int(math.ceil(len(objects) / float(NUM_BATCHES))),
                                                    max_q_size=1, workers=1)
    # Save probabilities
    entity.objects_probs = objects_probes

    ## Get labels
    # Get the max argument
    index_labels_per_sample = np.argmax(objects_probes, axis=1)
    # Get the object labels on hot vector per object [len(objects), 150]
    objects_labels = np.eye(len(hierarchy_mapping_objects), dtype='uint8')[index_labels_per_sample.reshape(-1)]
    # Save labels
    entity.objects_labels = objects_labels

    ## Get object features
    features_lst = []
    # Define the function
    for object in objects:
        try:
            img = get_img(url_data)
            # Get the mask: a dict with {x1,x2,y1,y2}
            mask_object = get_mask_from_object(object)
            # Saves as a box
            object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
            patch = img[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2], :]
            resized_img = get_img_resize(patch, config.crop_width, config.crop_height, type=config.padding_method)
            resized_img = np.expand_dims(resized_img, axis=0)
            get_features_output_func = K.function([predict_model.layers[0].input], [predict_model.layers[-2].output])
            features_model = get_features_output_func([resized_img])[0]
            features_lst.append(features_model)
        except Exception as e:
            logger.log("Exception for object: {0}, image: {1}".format(object, url_data))
            logger.log(str(e))
            traceback.print_exc()

    # Get the object features [len(objects), 2048]
    objects_features = np.concatenate(features_lst)
    # Save features
    entity.objects_features = objects_features


def predict_predicates_for_module(entity, objects, url_data, hierarchy_mapping_predicates):
    """
    This function predicts predicates for module later - predicates_probes [n, n, 51], predicates_features [n, n, 2048],
    predicates_labels [n, n, 51]
    :param objects: List of objects
    :param entity: Entity visual genome class
    :param url_data: a url data
    :param hierarchy_mapping_predicates: hierarchy_mapping for predicates
    :return:
    """

    # Create object pairs
    # Maybe list(itertools.permutations(objects, repeat=2))
    objects_pairs = list(itertools.product(objects, repeat=2))
    # Create a dict with key as pairs - (subject, object) and their values are predicates
    relations_dict = {}
    for relation in entity.relationships:
        relations_dict[(relation.subject.names[0], relation.object.names[0])] = relation.predicate

    # Create a data generator for VisualGenome for PREDICATES
    data_gen_val_predicates_vg = visual_genome_data_predicate_pairs_generator_with_batch(data=objects_pairs,
                                                                                         relations_dict=relations_dict,
                                                                                         hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                         config=config,
                                                                                         mode='validation',
                                                                                         batch_size=NUM_BATCHES,
                                                                                         evaluate=True)
    # Get the Predicate probabilities [n, 51]
    predicates_probes = predict_model.predict_generator(data_gen_val_predicates_vg,
                                                    steps=int(math.ceil(len(objects_pairs) / float(NUM_BATCHES))),
                                                    max_q_size=1, workers=1)
    # Reshape the predicates probabilites [n, n, 51]
    reshaped_predicates_probes = predicates_probes.reshape((len(objects), len(objects), len(hierarchy_mapping_predicates)))
    # Save probabilities
    entity.predicates_probes = reshaped_predicates_probes

    ## Get labels
    # Get the max argument
    index_labels_per_sample = np.argmax(predicates_probes, axis=1)
    # Get the object labels on hot vector per object [len(objects), 51]
    predicates_labels = np.eye(len(hierarchy_mapping_predicates), dtype='uint8')[index_labels_per_sample.reshape(-1)]
    # Reshape the predicates labels [n, n, 51]
    reshaped_predicates_labels = predicates_labels.reshape((len(objects), len(objects), len(hierarchy_mapping_predicates)))
    # Save labels
    entity.predicates_labels = reshaped_predicates_labels

    ## Get object features
    features_lst = []
    # Define the function
    for object_pair in objects_pairs:
        try:
            # Get Image
            img = get_img(url_data)
            # Get Subject and Object
            subject = object_pair[0]
            object = object_pair[1]
            # Calc Union-Box
            # Get the Subject mask: a dict with {x1,x2,y1,y2}
            mask_subject = get_mask_from_object(subject)
            # Saves as a box
            subject_box = np.array([mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])

            # Get the Object mask: a dict with {x1,x2,y1,y2}
            mask_object = get_mask_from_object(object)
            # Saves as a box
            object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])

            # Get the UNION box: a BOX (numpy array) with [x1,x2,y1,y2]
            union_box = find_union_box(subject_box, object_box)

            patch = img[union_box[BOX.Y1]: union_box[BOX.Y2], union_box[BOX.X1]: union_box[BOX.X2], :]
            resized_img = get_img_resize(patch, config.crop_width, config.crop_height, type=config.padding_method)
            resized_img = np.expand_dims(resized_img, axis=0)
            get_features_output_func = K.function([predict_model.layers[0].input],
                                                  [predict_model.layers[-2].output])
            features_model = get_features_output_func([resized_img])[0]
            features_lst.append(features_model)
        except Exception as e:
            logger.log("Exception for object: {0}, image: {1}".format(object_pair, url_data))
            logger.log(str(e))
            traceback.print_exc()

    # Get the object features [len(objects), 2048]
    predicates_features = np.concatenate(features_lst)
    # Number of features
    number_of_features = predicates_features.shape[1]
    # Reshape the predicates labels [n, n, 2048]
    reshaped_predicates_features = predicates_features.reshape((len(objects), len(objects), number_of_features))
    # Save features
    entity.predicates_features = reshaped_predicates_features


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

    # Get time and date
    time_and_date = get_time_and_date()

    # Path for the training folder
    path = os.path.join(PREDICATED_FEATURES_PATH, time_and_date)
    # Create a new folder for training
    create_folder(path)

    # Load detections dtype numpy array and hierarchy mappings
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "mini_filtered_data",
                                                                                          category='entities')
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
    number_of_classes_predicates = len(hierarchy_mapping_predicates)

    # Get the object and predicate model
    object_model = get_model(number_of_classes_objects, weight_path=objects_model_weight_path, config=config)
    predict_model = get_model(number_of_classes_predicates, weight_path=predicates_model_weight_path, config=config)

    logger.log('Starting Prediction')
    predicated_entities = []

    entities = entities[:1]

    # Predict each entity
    for entity in entities:
        logger.log('Predicting image id {0}'.format(entity.image.id))
        # Get the url image
        url_data = entity.image.url

        # Create Objects Mapping type
        objects = from_object_to_objects_mapping(entity.objects, hierarchy_mapping_objects, url_data)

        # Predict objects per entity
        predict_objects_for_module(entity, objects, url_data, hierarchy_mapping_objects)

        # Predict predicates per entity
        predict_predicates_for_module(entity, objects, url_data, hierarchy_mapping_predicates)

        predicated_entities.append(entity)

    logger.log('Finished Predicting entities')

    # Save entities
    logger.log("Saving Predicated entities")
    save_files(path, predicated_entities)
    logger.log("Finished successfully saving predicated_detections")
