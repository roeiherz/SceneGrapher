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

    # get data
    filtered_data, nof_predicates = prepare_data(1000)
    training_data, test_data = filtered_data.split(0.98)
    # embedded words module
    embed = WordEmbd()

    # create LangModule
    lang = LangModule(nof_predicates, embed.vector_dim)

    # get weights
    weights = lang.get_weights()

    # train
    sgd.sgd(lambda x: lang_module_sgd_wrapper(x, training_data, lang),
            weights,
            test_func=lambda x: lang_module_sgd_test(x, test_data, lang))

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


def lang_module_sgd_test(x, data, lang):
    """
    Wrapper for SGD training
    :param x: module parameters
    :param data: list of two objects of Data
    :param lang: the langauge module
    :return: cost and gradient of batch
    """
    f = lang.predict(data.worda, data.wordb, x)
    #predicate_ids = np.argmax(f, axis=0)
    correct_ans = np.sum(f[np.arange(len(data.worda)), data.predicate_ids])
    print("accuracy {0}".format(str(float(correct_ans) / np.sum(f))))


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
    while np.sum(np.logical_and(np.logical_and(R1.worda == R2.wordb, R1.predicate_ids == R2.predicate_ids), R1.wordb == R2.wordb)):
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
    lang = LangModule(nof_predicates, embed.vector_dim)

    # get weights
    params = lang.get_weights()

    gradcheck_naive(lambda x:
                    lang.cost_and_gradient(x, data[0], data[1]), params)


if __name__ == "__main__":
    train()
    # sanity_check()
