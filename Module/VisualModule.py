import numpy as np
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense
from numpy.core.umath_tests import inner1d
import os
import cPickle
import sys
from DesignPatterns.Detections import Detections
from keras_frcnn.Lib.Config import Config
from keras_frcnn.Lib.VisualGenomeDataGenerator import visual_genome_data_parallel_generator, get_img
from keras import backend as K
from keras.models import Model
from keras_frcnn.Lib.Zoo import ModelZoo
from keras_frcnn.Utils.Boxes import BOX
from keras_frcnn.Utils.Utils import get_img_resize

VG_VisualModule_PICKLES_PATH = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/VisualModule/Data/VisualGenome/"
OBJECTS_TRAINING_PATH = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Training/TrainingObjectsCNN/"
PREDICATES_TRAINING_PATH = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Training/TrainingPredicatesCNN/"
WEIGHTS_NAME = 'model_vg_resnet50.hdf5'


class VisualModule(object):
    """
    Visual module for scene-grapher
    """

    def __init__(self, objects_training_dir_name=OBJECTS_TRAINING_PATH, predicates_training_dir_name=PREDICATES_TRAINING_PATH):
        self.objects_model_weight_path = os.path.join(OBJECTS_TRAINING_PATH, objects_training_dir_name, WEIGHTS_NAME)
        self.predicates_model_weight_path = os.path.join(PREDICATES_TRAINING_PATH, predicates_training_dir_name, WEIGHTS_NAME)

        # Get argument
        if len(sys.argv) < 2:
            # Default GPU number
            gpu_num = 0
        else:
            # Get the GPU number from the user
            gpu_num = sys.argv[1]

        # Load class config
        self.config = Config(gpu_num)

        # Define GPU training
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_num)

        # Get the whole detections
        self.full_detections = self._get_detections(detections_file_name="full_filtered_detections.p")
        # Get the hierarchy mapping objects
        self.hierarchy_mapping_objects = cPickle.load(open("hierarchy_mapping_objects.p"))
        # Get the hierarchy mapping predicates
        self.hierarchy_mapping_predicates = cPickle.load(open("hierarchy_mapping_predicates.p"))
        # Set the number of classes
        self.number_of_classes = len(self.hierarchy_mapping_objects)
        # Get the object and predicate model
        self.object_model = self.get_model(self.number_of_classes, weight_path=self.objects_model_weight_path)
        # self.predict_model = self.get_model(self.number_of_classes, weight_path=self.predicates_model_weight_path)
        self.predict_model = None

    def extract_features(self, relation_ids):
        """
        extract visual features
        :param relation_ids: array of relationships ids
        :return: predicate_features, subject_probabilities, object_probabilities
        """

        # Sorted detections by their relation_ids
        detections = self.full_detections[relation_ids]

        # Define the generator
        data_gen_validation_vg = visual_genome_data_parallel_generator(data=detections,
                                                                       hierarchy_mapping=self.hierarchy_mapping_objects,
                                                                       config=self.config, mode='valid')

        # Get probabilities
        probes = self.object_model.predict_generator(data_gen_validation_vg, steps=len(detections) * 2, max_q_size=1,
                                                     workers=1)

        # Slice the Subject prob. (even index) [nof_samples, 150]
        subject_probabilities = probes[::2]

        # Slice the Object prob. (odd index) [nof_samples, 150]
        object_probabilities = probes[1::2]

        # Fill detections with Subject and Object probabilities - for future use
        # self.fill_detections_with_probes(probes, detections)

        # Get the features
        # Create matrix with a resize union box from all detections
        resized_union_box_mat = self.get_resize_images_array(detections)
        # Define a Graph function to extract the features from Global Average Pooling layer
        get_features_output = K.function([self.predict_model.layers[0].input], [self.predict_model.layers[-2].output])
        # Predict the Graph function [nof_samples, 2048]
        predicate_features = get_features_output([resized_union_box_mat])[0]

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
        predicate_likelihoods = inner1d(z[predicate_ids], predicate_features) + s[predicate_ids].flatten()
        subject_likelihoods = subject_probabilities[subject_ids]
        object_likelihoods = object_probabilities[object_ids]

        likelihoods = subject_likelihoods * predicate_likelihoods * object_likelihoods

        return likelihoods

    def _get_detections(self, detections_file_name="full_filtered_detections.p"):
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

    def _get_detections_by_ids(self, full_detections, relation_ids):
        """
        This function return the sorted detections by their relation_ids
        :param full_detections: full detections numpy dtype array
        :param relation_ids: list of relation_ids
        :return: detections numpy dtype array sorted by their relation_ids
        """

        # Find the indices of the detections that
        indices = np.searchsorted(full_detections[Detections.Id], relation_ids)
        detections = full_detections[indices]
        print 'debug'

    def get_model(self, number_of_classes, weight_path):
        """
        This function loads the model
        :param weight_path: model weights path
        :type number_of_classes: number of classes
        :return: model
        """

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

    def fill_detections_with_probes(self, probes, detections):
        """
        This function fill the detections with the Subject and Object probabilities
        :param probes: probabilities from the model predict
        :param detections: detections dtype numpy array
        """
        # Get the max probes for each sample
        probes_per_sample = np.max(probes, axis=1)
        # Slice the Subject prob (even index)
        detections[Detections.SubjectConfidence] = probes_per_sample[::2]
        # Slice the Object prob (odd index)
        detections[Detections.ObjectConfidence] = probes_per_sample[1::2]
        # Get the max argument
        index_labels_per_sample = np.argmax(probes, axis=1)

        # Get the inverse-mapping: int id to str label
        index_to_label_mapping = {label: id for id, label in self.hierarchy_mapping_objects.iteritems()}
        labels_per_sample = np.array([index_to_label_mapping[label] for label in index_labels_per_sample])

        # Slice the predicated Subject id (even index)
        detections[Detections.PredictSubjectClassifications] = labels_per_sample[::2]
        # Slice the predicated Object id (odd index)
        detections[Detections.PredictObjectClassifications] = labels_per_sample[1::2]

    def get_resize_images_array(self, detections):
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
            resized_img = get_img_resize(patch, self.config.crop_width, self.config.crop_height, type=self.config.padding_method)
            resized_img_lst.append(resized_img)

        return np.array(resized_img_lst)


if __name__ == '__main__':
    # Example
    tt = VisualModule(objects_training_dir_name="Sat_May_27_18:25:10_2017_full",
                      predicates_training_dir_name="Wed_May_31_16:19:26_2017")
    tt.extract_features(relation_ids=[1, 2, 15, 5, 25, 10])
