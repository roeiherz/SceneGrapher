from WordEmbd import WordEmbd
from Module import Module
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

    # get data
    filtered_data, nof_predicates = prepare_data(1000)
    training_data, test_data = filtered_data.split(0.98)
    # embedded words module
    embed = WordEmbd()

    # create LangModule
    nof_objects = 150
    visual_embed_size = 1
    module = Module(nof_objects, nof_predicates, embed.vector_dim, visual_embed_size)

    # get weights
    weights = module.get_params()

    # train
    sgd.sgd(lambda x: module_sgd_wrapper(x, training_data, module),
            weights,
            test_func=lambda x: module_sgd_test(x, test_data, module))

    return module


def module_sgd_wrapper(x, data, module):
    """
    Wrapper for SGD training
    :param x: module parameters
    :param data: list of two objects of Data
    :param module: the trained module
    :return: cost and gradient of batch
    """
    batch_size = 1
    cost = 0.0
    grad = np.zeros(x.shape)

    for i in xrange(batch_size):
        batch = get_random_data(data)
        cost_i, grad_i = module.get_gradient_and_loss(x, batch[0], batch[1])
        cost += cost_i / batch_size
        grad += grad_i / batch_size

    return cost, grad


def module_sgd_test(x, data, module):
    """
    Wrapper for SGD training
    :param x: module parameters
    :param data: list of two objects of Data
    :param module: the language module
    :return: cost and gradient of batch
    """
    predict = module.predict(data.worda, data.wordb, x)
    #predicate_ids = np.argmax(predict, axis=0)
    correct_ans = np.sum(predict[np.arange(len(data.worda)), data.predicate_ids])
    print("accuracy {0}".format(str(float(correct_ans) / np.sum(predict))))


def get_random_data(data, batch_size=1000):
    """
    Randomly select batch from the data
    TBD..
    :param batch_size:
    :param data: list of two objects of Data
    :return: list of two objects of Data (batch size)
    """
    batch = []

    indices1 = np.random.randint(0, data.worda.shape[0], batch_size)
    R1 = data.get_subset(indices1)
    indices2 = np.random.randint(0, data.worda.shape[0], batch_size)
    R2 = data.get_subset(indices2)
    # make sure no relation comapred to itself
    while np.sum(np.logical_and(np.logical_and(R1.worda == R2.worda, R1.predicate_ids == R2.predicate_ids), R1.wordb == R2.wordb)):
        indices2 = np.random.randint(0, data.worda.shape[0], batch_size)
        R2 = data.get_subset(indices2)

    return [R1, R2]


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    # embedded words module
    embed = WordEmbd()

    # get data - TBD
    data, nof_predicates = prepare_data(10)
    data = get_random_data(data)

    # create LangModule
    nof_objects = 150
    visual_embed_size = 10
    module = Module(nof_objects, nof_predicates, embed.vector_dim, visual_embed_size)

    # get weights
    params = module.get_params()

    gradcheck_naive(lambda x:
                    module.get_gradient_and_loss(x, data[0], data[1]), params)


if __name__ == "__main__":
    train()
    #sanity_check()