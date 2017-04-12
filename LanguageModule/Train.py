from WordEmbd import WordEmbd
from LangModule import LangModule
import sgd
import numpy as np

def train():
    """

    :return:
    """
    #embedded words module
    embed = WordEmbd()

    #create LangModule
    lang = LangModule(70, embed.vector_dim)

    #get data - TBD
    data = [Data("a", "b", 0, 5), Data("c", "d", 1, 3)]

    #get weights
    weights = lang.get_weights()

    #train
    sgd.sgd(lambda x: lang_module_sgd_wrapper(x, data, lang),
            weights)

def lang_module_sgd_wrapper(x, data, lang):

    batch_size = 100
    cost = 0.0
    grad = np.zeros(x.shape)

    for i in xrange(batch_size):
        batch = get_random_data(data)
        cost_i, grad_i = lang.cost_and_gradient(x, batch[0], batch[1])
        cost += cost_i / batch_size
        grad += grad_i / batch_size

    return cost, grad

def get_random_data(data):
    return data


class Data(object):
    def __init__(self, worda, wordb, predicate, instances):
        self.worda = worda
        self.wordb = wordb
        self.predicate = predicate
        self.instances = np.ones((1)) * instances

if __name__ == "__main__":
    train()