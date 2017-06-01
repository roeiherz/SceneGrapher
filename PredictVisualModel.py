from __future__ import print_function
from Data.VisualGenome.models import ObjectMapping, RelationshipMapping
from DesignPatterns.Detections import Detections
from keras_frcnn.Lib.VisualGenomeDataGenerator import visual_genome_data_generator, \
    visual_genome_data_parallel_generator, get_img, visual_genome_data_parallel_generator_with_batch
from keras_frcnn.Lib.Zoo import ModelZoo
from keras.applications.resnet50 import ResNet50
import os
import cPickle
import numpy as np
from keras_frcnn.Lib.Config import Config
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
import sys
import matplotlib.pyplot as plt

from keras_frcnn.Utils.Boxes import find_union_box, BOX
from keras_frcnn.Utils.Utils import VisualGenome_PICKLES_PATH, VG_VisualModule_PICKLES_PATH, get_mask_from_object, \
    get_img_resize, TRAINING_OBJECTS_CNN_PATH, TRAINING_PREDICATE_CNN_PATH, WEIGHTS_NAME
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
from keras_frcnn.Utils.Visualizer import VisualizerDrawer, CvColor
from keras_frcnn.Utils.data import get_sorted_data, generate_new_hierarchy_mapping, splitting_to_datasets, \
    process_to_detections

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2
NUM_EPOCHS = 1
NUM_BATCHES = 128

# If the allocation of training, validation and testing does not adds up to one
used_percent = TRAINING_PERCENT + VALIDATION_PERCENT + TESTING_PERCENT
if not used_percent == 1:
    error_msg = 'Data used percent (train + test + validation) is {0} and should be 1'.format(used_percent)
    print(error_msg)
    raise Exception(error_msg)

__author__ = 'roeih'


def preprocessing_objects(img_data, hierarchy_mapping, object_file_name='objects.p'):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param object_file_name: object pickle file name
    :param img_data: list of entities files
    :param hierarchy_mapping: dict of hierarchy_mapping
    :return: list of ObjectMapping
    """

    # Check if pickles are already created
    objects_path = os.path.join(VisualGenome_PICKLES_PATH, object_file_name)

    if os.path.isfile(objects_path):
        print('File is already exist {0}'.format(objects_path))
        objects = cPickle.load(file(objects_path, 'rb'))
        return objects

    # Get the whole objects from entities
    objects_lst = []
    correct_labels = hierarchy_mapping.keys()
    idx = 0
    for img in img_data:

        # Get the url image
        url = img.image.url
        # Get the objects per image
        objects = img.objects
        for object in objects:

            # Get the lable of object
            label = object.names[0]

            # Check if it is a correct label
            if label not in correct_labels:
                continue

            new_object_mapping = ObjectMapping(object.id, object.x, object.y, object.width, object.height, object.names,
                                               object.synsets, url)
            # Append the new objectMapping to objects_lst
            objects_lst.append(new_object_mapping)

        idx += 1
        print("Finished img: {}".format(idx))

    # Save the objects files to the disk
    objects_file = file(objects_path, 'wb')
    # Pickle objects_lst
    objects_array = np.array(objects_lst)
    cPickle.dump(objects_array, objects_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    objects_file.close()
    return objects_array


def preprocessing_relations(img_data, hierarchy_mapping, relation_file_name='relations.p'):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param relation_file_name: relation pickle file name
    :param img_data: list of entities files
    :param hierarchy_mapping: dict of hierarchy_mapping
    :return: list of RelationshipMapping
    """

    # Check if pickles are already created
    objects_path = os.path.join(VG_VisualModule_PICKLES_PATH, relation_file_name)

    if os.path.isfile(objects_path):
        print('File is already exist {0}'.format(objects_path))
        objects = cPickle.load(file(objects_path, 'rb'))
        return objects

    # Get the whole objects from entities
    objects_lst = []
    correct_labels = hierarchy_mapping.keys()
    idx = 0
    for img in img_data:

        # Get the url image
        url = img.image.url
        # Get the objects per image
        relations = img.relationships
        for relation in relations:

            # Get the label of object1 and object2
            label_o1 = relation.object.names[0]
            label_o2 = relation.subject.names[0]

            # Check if it is a correct label
            if label_o1 not in correct_labels or label_o2 not in correct_labels:
                continue

            new_relation_mapping = RelationshipMapping(relation.id, relation.subject, relation.predicate,
                                                       relation.object, relation.synset, url)
            # Append the new objectMapping to objects_lst
            objects_lst.append(new_relation_mapping)

        idx += 1
        print("Finished img: {}".format(idx))

    # Save the objects files to the disk
    objects_file = file(objects_path, 'wb')
    # Pickle objects_lst
    objects_array = np.array(objects_lst)
    cPickle.dump(objects_array, objects_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    objects_file.close()
    return objects_array


def get_classes_mapping_and_hierarchy_mapping_by_objects(objects):
    """
    This function creates classes_mapping and hierarchy_mapping by objects and updates the hierarchy_mapping accordingly
    :param objects: list of objects
    :return: dict of classes_mapping and hierarchy_mapping
    """
    classes_count_per_objects = {}
    hierarchy_mapping_per_objects = {}
    new_obj_id = 1
    for object in objects:
        # Get the lable of object
        label = object.names[0]

        # Update the classes_count dict
        if label in classes_count_per_objects:
            # Check if label is already in dict
            classes_count_per_objects[label] += 1
        else:
            # Init label in dict
            classes_count_per_objects[label] = 1

        # Update hierarchy_mapping dict
        if label not in hierarchy_mapping_per_objects:
            hierarchy_mapping_per_objects[label] = new_obj_id
            new_obj_id += 1
    return classes_count_per_objects, hierarchy_mapping_per_objects


def get_resize_images_array(detections):
    """
    This function calculates the resize image for each detection and returns a numpy ndarray
    :param detections: a numpy Detections dtype array
    :return: a numpy array of shape (len(detections), config.crop_width, config.crop_height , 3)
    """
    resized_img_lst = []
    for detection in detections:
        box = detection[Detections.UnionBox]
        url_data = detection[Detections.Url]
        img = get_img(url_data)
        patch = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
        resized_img = get_img_resize(patch, config.crop_width, config.crop_height, type=config.padding_method)
        resized_img_lst.append(resized_img)

    return np.array(resized_img_lst)


def load_full_detections(detections_file_name):
    """
    This function gets the whole filtered detections data (with no split between the  modules)
    :return: detections
    """
    # Check if pickles are already created
    detections_path = os.path.join(VG_VisualModule_PICKLES_PATH, detections_file_name)

    if os.path.isfile(detections_path):
        print('Detections numpy array is Loading from: {0}'.format(detections_path))
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
        print("Start loading Weights")
        model.load_weights(weight_path, by_name=True)
        print('Finished successfully loading weights from {}'.format(weight_path))

    except Exception as e:
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        raise Exception(e)

    print('Finished successfully loading Model')
    return model


if __name__ == '__main__':

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
        print("Object training folder parameter is: {}".format(objects_training_dir_name))
        predicates_training_dir_name = sys.argv[3]
        print("Predicate training folder parameter is: {}".format(predicates_training_dir_name))

    # Load class config
    config = Config(gpu_num)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)

    # classes_count, hierarchy_mapping, entities = get_sorted_data(classes_count_file_name="final_classes_count.p",
    #                                                              hierarchy_mapping_file_name="final_class_mapping.p",
    #                                                              entities_file_name="entities_example.p",
    #                                                              nof_labels=NOF_LABELS)
    #
    # # Get Visual Genome Data relations
    # relations = preprocessing_relations(entities, hierarchy_mapping, relation_file_name="relations.p")
    # # Process relations to numpy Detections dtype
    # detections = process_to_detections(relations, detections_file_name="detections.p")
    # # Split the data to train, test and validate
    # train_imgs, test_imgs, val_imgs = splitting_to_datasets(detections, training_percent=TRAINING_PERCENT,
    #                                                         testing_percent=TESTING_PERCENT, num_epochs=NUM_EPOCHS,
    #                                                         path=VG_VisualModule_PICKLES_PATH)

    # # todo: delete my data is only detections. should be change
    # detections = detections[:10]
    #
    # # Set new Hierarchy Mapping
    # new_hierarchy_mapping = generate_new_hierarchy_mapping(hierarchy_mapping)

    # Load detections dtype numpy array
    detections = load_full_detections(detections_file_name="mini_filtered_detections.p")

    detections = detections[:380]

    # Load hierarchy mappings
    # Get the hierarchy mapping objects
    hierarchy_mapping_objects = cPickle.load(open(os.path.join(VG_VisualModule_PICKLES_PATH,
                                                               "hierarchy_mapping_objects.p")))
    # Get the hierarchy mapping predicates
    hierarchy_mapping_predicates = cPickle.load(open(os.path.join(VG_VisualModule_PICKLES_PATH,
                                                                  "hierarchy_mapping_predicates.p")))

    # Check the training folders from which we take the weights aren't empty
    if not objects_training_dir_name or not predicates_training_dir_name:
        print("Error: No object training folder or predicate training folder has been given")
        exit()

    # Load the weight paths
    objects_model_weight_path = os.path.join(TRAINING_OBJECTS_CNN_PATH, objects_training_dir_name, WEIGHTS_NAME)
    predicates_model_weight_path = os.path.join(TRAINING_PREDICATE_CNN_PATH, predicates_training_dir_name, WEIGHTS_NAME)

    # Set the number of classes
    number_of_classes_objects = len(hierarchy_mapping_objects)
    number_of_classes_predicates = len(hierarchy_mapping_predicates)

    # # Create a data generator for VisualGenome
    # data_gen_validation_vg = visual_genome_data_parallel_generator(data=detections,
    #                                                                hierarchy_mapping=hierarchy_mapping_objects,
    #                                                                config=config, mode='valid')

    # Create a data generator for VisualGenome
    data_gen_validation_vg = visual_genome_data_parallel_generator_with_batch(data=detections,
                                                                              hierarchy_mapping=hierarchy_mapping_objects,
                                                                              config=config, mode='valid',
                                                                              batch_size=NUM_BATCHES)

    # Get the object and predicate model
    object_model = get_model(number_of_classes_objects, weight_path=objects_model_weight_path, config=config)
    predict_model = get_model(number_of_classes_predicates, weight_path=predicates_model_weight_path, config=config)

    # Make prediction
    # if K.image_dim_ordering() == 'th':
    #     input_shape_img = (3, None, None)
    # else:
    #     input_shape_img = (config.crop_height, config.crop_width, 3)
    #
    # img_input = Input(shape=input_shape_img, name="image_input")
    #
    # # Define ResNet50 model Without Top
    # net = ModelZoo()
    # model_resnet50 = net.resnet50_base(img_input, trainable=True)
    # model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
    # output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
    #     model_resnet50)
    #
    # # Define the model
    # model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
    # # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    # model.summary()
    #
    # # Load pre-trained weights for ResNet50
    # try:
    #     if config.load_weights:
    #         print("Start loading Weights")
    #         model.load_weights(config.base_net_weights, by_name=True)
    #         print('Finished successfully loading weights from {}'.format(weight_path))
    # except Exception as e:
    #     print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
    #         'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
    #         'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #     ))
    #     raise Exception(e)

    print('Starting Prediction')

    # The number of batches per epoch depends if size % batch_size == 0
    if len(detections) % NUM_BATCHES == 0:
        num_of_batches_per_epoch = len(detections) / NUM_BATCHES
    else:
        num_of_batches_per_epoch = len(detections) / NUM_BATCHES + 1

    # probes = object_model.predict_generator(data_gen_validation_vg, steps=len(detections) * 2, max_q_size=1, workers=1)
    probes = object_model.predict_generator(data_gen_validation_vg, steps=len(detections) / NUM_BATCHES, max_q_size=1,
                                            workers=1)
    # Slice the Subject prob (even index)
    # detections[Detections.SubjectConfidence] = probes[::2]
    detections[Detections.SubjectConfidence] = np.split(probes[::2], len(detections), axis=0)
    # Slice the Object prob (odd index)
    # detections[Detections.ObjectConfidence] = probes[1::2]
    detections[Detections.ObjectConfidence] = np.split(probes[1::2], len(detections), axis=0)
    # Get the max probes for each sample
    probes_per_sample = np.max(probes, axis=1)
    # Get the max argument
    index_labels_per_sample = np.argmax(probes_per_sample, axis=1)

    # Get the inverse-mapping: int id to str label
    index_to_label_mapping = {label: id for id, label in objects_model_weight_path.iteritems()}
    labels_per_sample = np.array([index_to_label_mapping[label] for label in index_labels_per_sample])

    # Slice the predicated Subject id (even index)
    detections[Detections.PredictSubjectClassifications] = labels_per_sample[::2]
    # Slice the predicated Object id (odd index)
    detections[Detections.PredictObjectClassifications] = labels_per_sample[1::2]

    # Get the Union-Box Features
    resized_img_mat = get_resize_images_array(detections)
    get_features_output = K.function([predict_model.layers[0].input], [predict_model.layers[-2].output])
    features_model = get_features_output([resized_img_mat])[0]
    detections[Detections.UnionFeature] = features_model

    print("Finished to predict probabilities and union features")
    print("Saving predicated_detections")
    # Save detections
    detections_filename = open(os.path.join(VG_VisualModule_PICKLES_PATH, "predicated_detections.p"), 'wb')
    # Pickle detections
    cPickle.dump(detections, detections_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    detections_filename.close()
    print("Finished successfully saving predicated_detections")

    # resized_img = np.expand_dims(resized_img, axis=0)
    # Get back the ResNet50 base part of a ResNet50 network trained on MS-COCO
    # output_model = ResNet50(weights=None, include_top=False)
    # output_model.load_weights(weights_path)
    # features_model = output_model.predict(resized_img_mat)

    # print('debug')

    # resized_img = np.expand_dims(resized_img, axis=0)
    # get_features_output = K.function([model.layers[0].input], [model.layers[-2].output])
    # features_model = get_features_output([resized_img_mat])[0]

    # tmp = model.layers[:]
    # model.layers = model.layers[-2]
    # Y = K.eval(model(K.variable(resized_img)))
    # model.layers = tmp[:]
    # print(Y)

    # output_model = Model(inputs=img_input, outputs=model.layers[-2].output)
    # x = tf.placeholder(tf.float32, shape=[224, 224, 3])
    # f = K.function([model.input], [model.layers[-2].output])
    # sess = tf.Session()
    # layer_output = f([x])[0]

    # Debug
    # if config.debug:
    #     ind = 0
    #     img_url = detections[0][Detections.Url]
    #     img = get_img(img_url)
    #     for i in range(len(detections)):
    #         new_img_url = detections[i][Detections.Url]
    #
    #         # img = get_img(new_img_url)
    #         if not img_url == new_img_url:
    #             cv2.imwrite("img {}.jpg".format(i), img)
    #             img = get_img(new_img_url)
    #             ind += 1
    #
    #         draw_subject_box = detections[i][Detections.SubjectBox]
    #         draw_object_box = detections[i][Detections.ObjectBox]
    #         VisualizerDrawer.draw_labeled_box(img, draw_subject_box,
    #                                           label=detections[i][Detections.SubjectClassifications] + "/" +
    #                                                 labels_per_sample[i],
    #                                           rect_color=CvColor.GREEN,
    #                                           scale=2000)
    #         VisualizerDrawer.draw_labeled_box(img, draw_object_box,
    #                                           label=detections[i][Detections.ObjectClassifications] + "/" +
    #                                                 labels_per_sample[i],
    #                                           rect_color=CvColor.RED,
    #                                           scale=2000)
    #         # cv2.imwrite("img {}.jpg".format(i), img)
    #     cv2.imwrite("img {}.jpg".format(ind), img)
    #     print('debug')
    # model.predict_on_batch()
