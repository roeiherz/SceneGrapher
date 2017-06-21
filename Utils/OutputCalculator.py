import abc

__author__ = 'roeih'


class OutputCalculator(object):
    @abc.abstractmethod
    def collect(self):
        pass

    @abc.abstractmethod
    def export(self):
        pass

    @abc.abstractmethod
    def draw(self):
        pass


class RelationCalculator(OutputCalculator):

    def __init__(self):
        self._summary_df = None

    def collect(self):
        pass

    def export(self):
        pass

    def draw(self):
        pass


if __name__ == '__main__':
    tt = RelationCalculator()
