from ModuleLogger import ModuleLogger
from WordEmbd import WordEmbd
from Module import Module
import sgd
import numpy as np
from data import *
from gradcheck import gradcheck_naive
import yaml
import inspect

def train(
    name = "none",
    iterations = 100000,
    start_iterations = 0,
    learning_rate = 0.01,
    learning_rate_steps = 100,
    test_steps = 100,
    coeff_k = 0.005,
    coeff_l = 0.02,
    coeff_reg_visual = 0.001,
    coeff_reg_lang = 0.001,
    saved_params_file_name = "best_params.npy",
    word_embed_size=50,
    visual_embed_size=2048):
    """
    Basic function to train language module.
    TBD: improve
    :return: trained language module
    """
    # create logger
    logger = ModuleLogger(name)

    # print train params
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    logger.log('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        logger.log("    %s = %s" % (i, values[i]))


    logger.log("Prepare Data")
    module_data = prepare_data()
    # get data
    training_data = module_data["train"]
    validation_data = module_data["validation"]

    # embedded words module
    logger.log("Load Embed Word")
    embed = WordEmbd(word_embed_size)

    # create Module
    logger.log("Create Module")
    object_ids = module_data["object_ids"]
    predicate_ids = module_data["predicate_ids"]
    module = Module(object_ids, predicate_ids, embed.vector_dim, visual_embed_size)

    # get weights
    weights = module.get_params()

    # train
    logger.log("Train")
    sgd.sgd(lambda x: module_sgd_wrapper(x, training_data, module, coeff_k, coeff_l, coeff_reg_visual, coeff_reg_lang),
            weights,
            test_func=lambda x: module_sgd_test(x, validation_data, module, training_data),
            step=learning_rate,
            iterations=iterations,
            anneal_every=learning_rate_steps,
            test_every=test_steps,
            start_iterations=start_iterations,
            saved_params_file_name=saved_params_file_name)

    return module



def module_sgd_wrapper(x, data, module, coeff_k, coeff_l, coeff_reg_visual, coeff_reg_lang):
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
        cost_i, grad_i = module.get_gradient_and_loss(x, batch[0], batch[1],
                                                      coeff_k=coeff_k, coeff_l=coeff_l,
                                                      coeff_reg_visual=coeff_reg_visual,
                                                      coeff_reg_lang=coeff_reg_lang)
        cost += cost_i / batch_size
        grad += grad_i / batch_size

    return cost, grad


def module_sgd_test(x, test_data, module, train_data):
    """
    Wrapper for SGD training
    :param x: module parameters
    :param data: list of two objects of Data
    :param module: the language module
    :return: cost and gradient of batch
    """
    module_sgd_test_data(x, test_data, module, "test")
    module_sgd_test_data(x, train_data, module, "train")


def module_sgd_test_data(x, data, module, name):
    predict, acc_percent = module.predict(data, x)
    correct_ans = 0
    for index in range(len(data.worda)):
        if predict[index][0] == data.subject_ids[index] and predict[index][1] == data.predicate_ids[index] and predict[index][2] == data.object_ids[index]:
            correct_ans += 1

    logger = ModuleLogger()
    logger.log("{0} accuracy {1} acc percent {2}".format(name, str(float(correct_ans) / len(predict)), str(float(np.sum(acc_percent)) / len(acc_percent))))

def get_random_data(data, batch_size=128):
    """
    Randomly select batch from the data
    TBD..
    :type data: Data
    :param batch_size:
    :param data: list of two objects of Data
    :return: list of two objects of Data (batch size)
    """
    batch = []

    # indices1 = np.random.randint(0, data.worda.shape[0], batch_size)
    indices1 = np.random.choice(data.worda.shape[0], batch_size, replace=False)

    R1 = data.get_subset(indices1)
    # indices2 = np.random.randint(0, data.worda.shape[0], batch_size)
    # indices2 = np.random.choice(data.worda.shape[0], batch_size, replace=False)
    R2 = data.get_random(batch_size)
    # make sure no relation compared to itself
    while np.sum(np.logical_and(np.logical_and(R1.worda == R2.worda, R1.predicate_ids == R2.predicate_ids), R1.wordb == R2.wordb)):
        # indices2 = np.random.randint(0, data.worda.shape[0], batch_size)
        R2 = data.get_random(batch_size)
        #indices2 = np.random.choice(data.worda.shape[0], batch_size, replace=False)
        #R2 = data.get_subset(indices2)

    return [R1, R2]


def sanity_check(word_embed_size=50, visual_embed_size=2048):
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    # get data
    module_data = prepare_data()
    training_data = module_data["train"]
    batch = get_random_data(training_data)
    test_data = module_data["test"]
    # embedded words module
    embed = WordEmbd(word_embed_size)

    # create LangModule
    object_ids = module_data["object_ids"]
    predicate_ids = module_data["predicate_ids"]
    module = Module(object_ids, predicate_ids, embed.vector_dim, visual_embed_size)

    # get weights
    params = module.get_params()

    gradcheck_naive(lambda x:
                    module.get_gradient_and_loss(x, batch[0], batch[1]), params)


if __name__ == "__main__":
    #train()
    #sanity_check()
    from multiprocessing import Process
    stream = file('params.yaml', 'r')
    params = yaml.load(stream)

    nof_processes = params["nof_p"]

    processes = []
    for process in range(1, nof_processes + 1):
        process_params = params[process]
        name = process_params["name"]
        learning_rate = process_params["learning_rate"]
        learning_rate_steps = process_params["learning_rate_steps"]
        iterations = process_params["iterations"]
        test_steps = process_params["test_steps"]
        coeff_k = process_params["coeff_k"]
        coeff_l = process_params["coeff_l"]
        coeff_reg_visual = process_params["coeff_reg_visual"]
        coeff_reg_lang = process_params["coeff_reg_lang"]
        saved_params_file_name = process_params["saved_params_file_name"]
        start_iterations = process_params["start_iterations"]
        word_embed_size = process_params["word_embed_size"]
        p = Process(target=train, args=(
        name, iterations, start_iterations, learning_rate, learning_rate_steps, test_steps, coeff_k,
        coeff_l, coeff_reg_visual, coeff_reg_lang, saved_params_file_name, word_embed_size))
        p.start()
        processes.append(p)

    # wait until all processes done
    for p in processes:
        p.join()