import abc

__author__ = 'roeih'


class DataSetGenerator(object):
    """
    this class represents an abstract DataSet
    """

    @abc.abstractmethod
    def __init__(self, name, visualize=False):
        self.name = name
        self._visualize = visualize

    @abc.abstractmethod
    def get_data(self, input_path):
        pass

    def __str__(self):
        return '{0}, '.format(self.name)