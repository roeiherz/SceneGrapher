from WordEmbd import WordEmbd
from LangModule import LangModule
import sgd
import numpy as np
from data import *
from gradcheck import gradcheck_naive

def train():
    """
    Basic function to train language module.
    TBD: improve
    :return: trained langauge module
    """

    #get data - TBD
    R1, R2, nof_predicates = prepare_data(1000)
    data = [R1, R2]

    #embedded words module
    embed = WordEmbd()

    # create LangModule
    lang = LangModule(nof_predicates, embed.vector_dim)

    #get weights
    weights = lang.get_weights()

    #train
    sgd.sgd(lambda x: lang_module_sgd_wrapper(x, data, lang),
            weights)

    return lang

def lang_module_sgd_wrapper(x, data, lang):
    """
    Wrapper for SGD training
    :param x: module parameters
    :param data: list of two objects of Data
    :param lang: the langauge module
    :return: cost and gradient of batch
    """
    batch_size = 1
    cost = 0.0
    grad = np.zeros(x.shape)

    for i in xrange(batch_size):
        batch = get_random_data(data)
        cost_i, grad_i = lang.cost_and_gradient(x, batch[0], batch[1])
        cost += cost_i / batch_size
        grad += grad_i / batch_size

    return cost, grad

def get_random_data(data):
    """
    Randomly select batch from the data
    TBD..
    :param data: list of two objects of Data
    :return: list of two objects of Data (batch size)
    """
    return data



def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    # embedded words module
    embed = WordEmbd()

    # get data - TBD
    R1, R2, nof_predicates = prepare_data(10)
    data = [R1, R2]

    # create LangModule
    lang = LangModule(nof_predicates, embed.vector_dim)

    # get weights
    params = lang.get_weights()

    gradcheck_naive(lambda x:
                    lang.cost_and_gradient(x, data[0], data[1]), params)

if __name__ == "__main__":
    train()
    #sanity_check()