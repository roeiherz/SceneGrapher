from numpy.core.umath_tests import inner1d
import os
import cPickle
from DesignPatterns.Detections import Detections


VG_VisualModule_PICKLES_PATH = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/VisualModule/Data/VisualGenome/"


class VisualModule(object):
    """
    Visual module for scene-grapher
    """

    def __init__(self):
        # Get the whole detections
        # self.full_detections = self.get_detections(detections_file_name="predicated_mini_fixed_detections.p")
        self.full_detections = self.get_detections(detections_file_name="predicated_mini_fixed_detections_probes.p")

    def extract_features(self, relation_ids):
        """
        extract visual features and object and subject probabilities
        :param relation_ids: array of relationships ids
        :return: predicate_features, subject_probabilities, object_probabilities
        """

        # Sorted detections by their relation_ids
        indx = np.where(np.in1d(list(detections[Detections.Id]), relation_ids) == True)
        detections = self.full_detections[indx]

        # Check if loading detections succeed
        if detections is None:
            print("Error: detections wan't loaded")
            return None, None, None

        # Subject prob. [nof_samples, 150]
        subject_probabilities = detections[Detections.SubjectConfidence]

        # Object prob. [nof_samples, 150]
        object_probabilities = detections[Detections.ObjectConfidence]

        # Features [nof_samples, 2048]
        predicate_features = detections[Detections.UnionFeature]

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

if __name__ == '__main__':
    # Example
    tt = VisualModule()
    tt.extract_features(relation_ids=[1, 2, 15, 5, 25, 10])
