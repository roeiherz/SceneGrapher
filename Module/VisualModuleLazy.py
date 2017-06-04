from numpy.core.umath_tests import inner1d
import os
import cPickle
import sys
sys.path.append("..")
from DesignPatterns.Detections import Detections
import numpy as np
from Utils.Utils import softmax

VG_VisualModule_PICKLES_PATH = "../VisualModule/Data/VisualGenome/"
#VG_VisualModule_PICKLES_PATH = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/VisualModule/Data/VisualGenome/"


class VisualModule(object):
    """
    Visual module for scene-grapher
    """

    def __init__(self, evaluate=False):
        # Get the whole detections
        # self.full_detections = self.get_detections(detections_file_name="predicated_mini_fixed_detections.p")
        self.full_detections = self.get_detections(detections_file_name="predicated_mini_fixed_detections_url.p")
        self.evaluate = evaluate

    def extract_features(self, relation_ids):
        """
        extract visual features and object and subject probabilities
        :param relation_ids: array of relationships ids
        :return: predicate_features, subject_probabilities, object_probabilities
        """
        # Sorted detections by their relation_ids
        indx = np.where(np.in1d(list(self.full_detections[Detections.Id]), relation_ids) == True)
        detections = self.full_detections[indx]

        # Check if loading detections succeed
        if len(detections) != len(relation_ids):
            print("Error: not all detections was found")
            ee = np.where(np.in1d(relation_ids, list(self.full_detections[Detections.Id])) == False)
            print(relation_ids[ee])

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

    def get_detections(self, detections_file_name="predicated_mini_detections.p"):
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

    def predicate_predict(self, predicate_features, z, s):
        """
            predict a probability per predicate given visual module params
            :param predicate_features: visual features extracted
            :param z: visual module params
            :param s: visual params
            :return: probability per predicate
            """
        predicate_likelihoods = np.dot(z, predicate_features.T).T + s.flatten().T
        #predicate_probability = softmax(predicate_likelihoods)

        return predicate_likelihoods

    def extract_features_for_evaluate(self, objects, subjects):
        """
        This function
        :param objects: object which is a Object VisualGenome type
        :param subjects: subject which is a Object VisualGenome type
        :return: 
        """
        pass


if __name__ == '__main__':
    # Example
    tt = VisualModule()
    tt.extract_features(relation_ids=np.array([355181, 355198, 355216, 5, 25, 10]))
