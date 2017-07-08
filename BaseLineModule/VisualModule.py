from __future__ import print_function
import sys
import cv2

sys.path.append("..")
from Utils.Logger import Logger
from FilesManager.FilesManager import FilesManager

from FeaturesExtraction.Utils.Visualizer import CvColor, VisualizerDrawer
from FeaturesExtraction.Lib.VisualGenomeDataGenerator import visual_genome_data_parallel_generator_with_batch, \
    visual_genome_data_cnn_generator_with_batch
from numpy.core.umath_tests import inner1d
import os
from FeaturesExtraction.Lib.Config import Config
from FeaturesExtraction.Utils.Utils import TRAINING_OBJECTS_CNN_PATH, \
    TRAINING_PREDICATE_CNN_PATH, WEIGHTS_NAME, get_mask_from_object, get_img
from DesignPatterns.Detections import Detections
import numpy as np
from Utils.Utils import softmax, get_detections
from keras import backend as K
from keras.models import Model
from FeaturesExtraction.Lib.Zoo import ModelZoo
from FeaturesExtraction.Utils.Boxes import BOX, find_union_box
from FeaturesExtraction.Utils.Utils import get_img_resize
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense


class VisualModule(object):
    """
    Visual module for scene-grapher
    """

    def __init__(self):
        """
        Initialize class visual module
        :param objects_training_dir_name: objects training dir name for taking the weights
        :param predicates_training_dir_name: predicates training dir name for taking the weights
        """
        # get logger
        logger = Logger()

        # get files manager
        filesmanager = FilesManager()

        # Get the whole detections
        # self.full_detections = get_detections(detections_file_name="predicated_mini_fixed_detections_url.p")
        self.full_detections = get_detections()

        # Get Mapping dict between Detections.Id to index
        self.mapping_id_to_ind_dict = {}
        idx = 0
        for id in self.full_detections[Detections.Id]:
            self.mapping_id_to_ind_dict[id] = idx
            idx += 1

        # Check if loading detections succeed
        if self.full_detections is None:
            logger.log("Error: No detections have been found")
            raise Exception

        self.objects_model_weight_path = filesmanager.get_file_path("scene_graph_base_module.visual_module.object_cnn")
        self.predicates_model_weight_path = filesmanager.get_file_path("scene_graph_base_module.visual_module.predicate_cnn")

        # A flag which make sure we are initialized networks only once
        self.networks_initialization_flag = False
        # Parameters for the Networks
        self.config = None
        self.hierarchy_mapping_objects = None
        self.hierarchy_mapping_predicates = None
        self.objects_nof_classes = None
        self.predicates_nof_classes = None
        self.object_model = None
        self.predict_model = None
        self.gpu_num = None
        self.batch_size = None

    def extract_features(self, relation_ids):
        """
        extract visual features and object and subject probabilities
        :param relation_ids: array of relationships ids
        :return: predicate_features, subject_probabilities, object_probabilities
        """
        # get logger
        logger = Logger()

        # Sorted detections by their relation_ids
        detections_indx = [self.mapping_id_to_ind_dict[relation] for relation in relation_ids]
        detections = self.full_detections[detections_indx]

        # todo: remove after checking with testing
        # # Sorted detections by their relation_ids
        # detections_indx = np.zeros(len(relation_ids), dtype=int)
        # for indx in range(len(relation_ids)):
        #     detections_indx[indx] = \
        #         np.where(np.in1d(list(self.full_detections[Detections.Id]), relation_ids[indx]) == True)[0][0]
        # detections = self.full_detections[detections_indx]
        #
        # # Sorted detections by their relation_ids
        # indx = np.where(np.in1d(list(self.full_detections[Detections.Id]), relation_ids) == True)
        # detections = self.full_detections[indx]

        # Check if loading detections succeed
        if len(detections) != len(relation_ids):
            logger.log("Error: not all detections was found")
            ee = np.where(np.in1d(relation_ids, list(self.full_detections[Detections.Id])) == False)
            logger.log(relation_ids[ee])

        # Subject prob. [nof_samples, 150]
        subject_probabilities = np.concatenate(detections[Detections.SubjectConfidence])

        # Object prob. [nof_samples, 150]
        object_probabilities = np.concatenate(detections[Detections.ObjectConfidence])

        # Features [nof_samples, 2048]
        predicate_features = np.concatenate(detections[Detections.UnionFeature])

        return predicate_features, subject_probabilities, object_probabilities

    def likelihood(self, subject_ids, object_ids, predicate_ids, predicate_features, subject_probabilities,
                   object_probabilities, z, s):
        """
        Get likelihoods of relationships given visual features and visual model parameters
        :param subject_ids: subject ids of relationships (DIM: batch_size X 1)
        :param object_ids: object ids of relationships (DIM: batch_size X 1)
        :param predicate_ids: predicate ids of relationships (DIM: batch_size X 1)
        :param predicate_features: predicate features extracted
        :param subject_probabilities: probability for each subject id
        :param object_probabilities: probability for each object id
        :param z: visual module parameters
        :param s: visual module parameters
        :return:
        """
        predicate_likelihoods = np.dot(z, predicate_features).flatten() + s.flatten()
        # predicate_prob = softmax(predicate_likelihoods)[predicate_ids]
        subject_prob = subject_probabilities[subject_ids]
        object_prob = object_probabilities[object_ids]

        likelihoods = subject_prob * predicate_likelihoods[predicate_ids] * object_prob

        return likelihoods, subject_prob, object_prob

    def predicate_predict(self, predicate_features, z, s):
        """
            predict a probability per predicate given visual module params
            :param predicate_features: visual features extracted
            :param z: visual module params
            :param s: visual params
            :return: probability per predicate
            """
        predicate_likelihoods = np.dot(z, predicate_features.T).T + s.flatten().T
        # predicate_probability = softmax(predicate_likelihoods)

        return predicate_likelihoods

    def extract_features_for_evaluate(self, subject, object, img_url):
        """
        This function
        :param subject: subject which is a Object VisualGenome type
        :param object: object which is a Object VisualGenome type
        :param img_url: image url which the objects are taken from
        :return: 
        """
        # get logger
        logger = Logger()
        if self.hierarchy_mapping_objects is None or self.config is None:
            logger.log("Error: Networks didn't initialize correctly")

        # Add the url to the objects
        subject.url = img_url
        object.url = img_url

        # Create data images
        val_imgs = np.array([subject, object])
        data_gen_validation_vg = visual_genome_data_cnn_generator_with_batch(data=val_imgs,
                                                                             hierarchy_mapping=self.hierarchy_mapping_objects,
                                                                             config=self.config, mode='validation',
                                                                             batch_size=1)
        # Start prediction
        # print('Starting Prediction')
        # print('Predicting Probabilities')

        # Calculating Probabilities from objects [2 , 150]
        probes = self.object_model.predict_generator(data_gen_validation_vg, steps=2, max_q_size=1, workers=1)

        # Get Subject Probabilities - [1, 150]
        subject_probabilities = probes[0]

        # Get Object Probabilities - [1, 150]
        object_probabilities = probes[1]

        # print('Calculating Union-Box Features')
        # Define the function
        get_features_output_func = K.function([self.predict_model.layers[0].input],
                                              [self.predict_model.layers[-2].output])

        # Subject
        # Get the mask: a dict with {x1,x2,y1,y2}
        subject_mask = get_mask_from_object(subject)
        # Saves as a box
        subject_box = np.array([subject_mask['x1'], subject_mask['y1'], subject_mask['x2'], subject_mask['y2']])

        # Object
        # Get the mask: a dict with {x1,x2,y1,y2}
        object_mask = get_mask_from_object(subject)
        # Saves as a box
        object_box = np.array([object_mask['x1'], object_mask['y1'], object_mask['x2'], object_mask['y2']])

        # Find the union box
        union_box = find_union_box(subject_box, object_box)
        img = get_img(img_url)
        patch = img[union_box[BOX.Y1]: union_box[BOX.Y2], union_box[BOX.X1]: union_box[BOX.X2], :]
        resized_img = get_img_resize(patch, self.config.crop_width, self.config.crop_height,
                                     type=self.config.padding_method)
        resized_img = np.expand_dims(resized_img, axis=0)

        # Get the feature - [1, 2048]
        features_model = get_features_output_func([resized_img])[0]

        return subject_probabilities, object_probabilities, features_model

    def initialize_networks(self, gpu_num, batch_num=128):
        """
        This function initialize the networks only once
        :param gpu_num: GPU number
        :param batch_num: Number of batches which the networks will be run
        """
        # get logger
        logger = Logger()

        # get files manager
        filesmanager = FilesManager()

        # Check if we didn't already initialize networks once
        if self.networks_initialization_flag:
            logger.log("Error: We already initialized networks")
            raise Exception

        # Check that GPU number has been declared
        if self.gpu_num:
            logger.log("Error: No GPU number has been declared")
            raise Exception

        # Check if weights are declared properly
        if not os.path.exists(self.predicates_model_weight_path) or not os.path.exists(self.objects_model_weight_path):
            logger.log("Error: No Weights have been found")
            raise Exception

        # Set GPU number
        self.gpu_num = gpu_num

        # Set number of batches
        self.batch_size = batch_num

        # Load class config
        self.config = Config(self.gpu_num)

        # Define GPU training
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_num)

        # Define the hierarchy mapping objects and predicates paths

        # Get the hierarchy mapping objects
        self.hierarchy_mapping_objects = filesmanager.load_file("data.visual_genome.hierarchy_mapping_objects")
        # Get the hierarchy mapping predicates
        self.hierarchy_mapping_predicates = filesmanager.load_file("data.visual_genome.hierarchy_mapping_predicates")
        # Set the number of classes of object
        self.objects_nof_classes = len(self.hierarchy_mapping_objects)
        self.predicates_nof_classes = len(self.hierarchy_mapping_predicates)
        # Get the object and predicate model
        self.object_model = self.get_model(self.objects_nof_classes, weight_path=self.objects_model_weight_path)
        self.predict_model = self.get_model(self.predicates_nof_classes, weight_path=self.predicates_model_weight_path)

        # Set initialized_networks to True. This function will never run again
        self.networks_initialization_flag = True

    def get_model(self, number_of_classes, weight_path):
        """
        This function loads the model
        :param weight_path: model weights path
        :type number_of_classes: number of classes
        :return: model
        """
        # get logger
        logger = Logger()

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
        else:
            input_shape_img = (self.config.crop_height, self.config.crop_width, 3)

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

    def _debug_detections(self):
        """
        This function will print negative patches
        :param path: name path
        """
        # get files manager
        filesmanager = FilesManager()
        path = filesmanager.get_file_path("data.visual_genome.pics")

        # Get index of failed prediction of Subject
        indx = np.where((self.full_detections[Detections.SubjectClassifications] !=
                         self.full_detections[Detections.PredictSubjectClassifications]) |
                        (self.full_detections[Detections.ObjectClassifications] !=
                         self.full_detections[Detections.PredictObjectClassifications]))

        # Define the hierarchy mapping objects and predicates paths
        # Get the hierarchy mapping objects
        self.hierarchy_mapping_objects = filesmanager.load_file("data.visual_genome.hierarchy_mapping_objects")
        # Get the hierarchy mapping predicates
        self.hierarchy_mapping_predicates = filesmanager.load_file("data.visual_genome.hierarchy_mapping_predicates")

        for detection in self.full_detections:
            if (detection[Detections.SubjectClassifications] not in self.hierarchy_mapping_objects or
                        detection[Detections.SubjectClassifications] not in self.hierarchy_mapping_objects):
                print(detection)

        # Get the negatives
        negatives = self.full_detections[indx]
        img_url = negatives[0][Detections.Url]
        img = get_img(img_url)

        for negative in negatives:

            id = negative[Detections.Url].split("/")[-1]
            new_img_url = negative[Detections.Url]

            if not img_url == new_img_url:
                cv2.imwrite(path + "{}".format(id), img)
                img = get_img(new_img_url)

            if negative[Detections.SubjectClassifications] != negative[Detections.PredictSubjectClassifications]:
                draw_subject_box = negative[Detections.SubjectBox]
                VisualizerDrawer.draw_labeled_box(img, draw_subject_box,
                                                  label=negative[Detections.PredictSubjectClassifications] + "/" +
                                                        negative[Detections.SubjectClassifications],
                                                  rect_color=CvColor.BLACK, scale=500)

            if negative[Detections.ObjectClassifications] != negative[Detections.PredictObjectClassifications]:
                draw_object_box = negative[Detections.ObjectBox]
                VisualizerDrawer.draw_labeled_box(img, draw_object_box,
                                                  label=negative[Detections.PredictObjectClassifications] + "/" +
                                                        negative[Detections.ObjectClassifications],
                                                  rect_color=CvColor.BLUE, scale=500)

        cv2.imwrite(path + "{}".format(id), img)
