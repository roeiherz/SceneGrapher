import abc
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__author__ = 'roeih'


class DimensionReducer(object):
    """
    General Dimension Reducer
    """

    @abc.abstractmethod
    def _reduce(self, data, labels, goal_dimension):
        pass

    def reduce(self, data, labels, goal_dimension):
        """
        this function reduces the data's dimension
        :param data: array of the data
        :type data: numpy.ndarray
        :param labels: array of integers which labels the data
        :type labels: numpy.ndarray
        :param goal_dimension: the target dimension of reduction
        :type goal_dimension: int
        :return:
        """
        start_time = time.clock()
        data_reduced, reducer_fitted = self._reduce(data, labels, goal_dimension)
        end_time = time.clock()
        print('Reduction took total of {}s'.format(end_time - start_time))
        return data_reduced, reducer_fitted


class tSNEReducer(DimensionReducer):
    """
    t-sne Reducer
    """

    def __init__(self, n_iterations=1000, learning_rate=1000.0):
        self._n_iterations = n_iterations
        self._learning_rate = learning_rate

    def _reduce(self, data, labels, goal_dimension):
        print('Dimension reduction started using tSNE')
        tsne = TSNE(n_components=goal_dimension, random_state=0, n_iter=self._n_iterations,
                    learning_rate=self._learning_rate,
                    perplexity=15.0)
        tsne_output = tsne.fit_transform(data)
        return tsne_output, tsne


class PCAReducer(DimensionReducer):
    """
    PCA Reducer
    """

    def _reduce(self, data, labels, goal_dimension):
        print('Dimension reduction started using PCA')
        pca = PCA(n_components=goal_dimension)
        pca.fit(data)
        pca_output = pca.transform(data)
        return pca_output, pca


class tSNEusingPCA(DimensionReducer):
    """
    t-sne with using PCA reducer
    """
    def __init__(self, pca_goal_dimension, ica_n_iterations=1000, ica_learning_rate=1000.0):
        self._pca_goal_dimension = pca_goal_dimension
        self._ica_n_iterations = ica_n_iterations
        self._ica_learning_rate = ica_learning_rate

    def _reduce(self, data, labels, goal_dimension):
        pca_reducer = PCAReducer()
        print('Reduction using PCA started')
        pca_start_time = time.clock()
        pca_output, pca = pca_reducer.reduce(data, labels, self._pca_goal_dimension)
        pca_end_time = time.clock()
        print('PCA reduction ended, took {}s'.format(pca_end_time - pca_start_time))
        pca_variance = pca.explained_variance_ratio_
        print('pca total variance - {0}'.format(sum(pca_variance)))
        tsne_reducer = tSNEReducer(n_iterations=self._ica_n_iterations, learning_rate=self._ica_learning_rate)
        print('Second step - reduction using tSNE started')
        tsne_start_time = time.clock()
        tsne_output, tsne = tsne_reducer.reduce(pca_output, labels, goal_dimension)
        tsne_end_time = time.clock()
        print('tSNE reduction ended, took {}s'.format(tsne_end_time - tsne_start_time))

        return tsne_output, pca


