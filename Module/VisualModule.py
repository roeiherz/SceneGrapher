import numpy as np
from numpy.core.umath_tests import inner1d


class VisualModule(object):
    """
    Visual module for scene grapher
    """

    def __init__(self, nof_objects, visual_embed_size):
        self.nof_objects = nof_objects
        self.visual_embed_size = visual_embed_size

    def extract_features(self, relation_ids):
        """
        extract visual features
        :param relation_ids: array of relationships ids
        :return: predicate_features, subject_probabilities, object_probabilities
        """

        # todo: Herzig
        # TBD
        # [1000, 2048]
        predicate_features = np.ones((len(R1.worda), self.visual_embed_size))
        # [1000, 150]
        subject_probabilities = np.ones((len(R1.worda), self.nof_objects))
        object_probabilities = np.ones((len(R1.worda), self.nof_objects))

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
