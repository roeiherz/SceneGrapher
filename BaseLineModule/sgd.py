#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
import os

from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger

SAVE_PARAMS_EVERY = 100
import glob
import random
import os.path as op
import cPickle as pickle


def load_saved_params(saved_params_file_name, start_iterations):
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    filemanager = FilesManager()
    saved_params_path = filemanager.get_file_path("scene_graph_base_module.train.saved_params")
    saved_params_path = os.path.join(saved_params_path, saved_params_file_name)
    if saved_params_file_name != None and os.path.exists(saved_params_path):
        with open(saved_params_path, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
            iter = start_iterations
            return iter, params, state
    else:
        st = 0
        saved_params_path = filemanager.get_file_path("scene_graph_base_module.train.saved_params")
        files_name = os.path.join(saved_params_path, "saved_params_*.npy")
        for f in glob.glob(files_name):
            iter = int(op.splitext(op.basename(f))[0].split("_")[2])
            if (iter > st):
                st = iter

        if st > 0:
            with open(files_name % st, "r") as f:
                params = pickle.load(f)
                state = pickle.load(f)
            return st, params, state
        else:
            return st, None, None


def save_params(iter, params, dir):
    filemanager = FilesManager()
    saved_params_path = filemanager.get_file_path("scene_graph_base_module.train.saved_params")
    saved_params_path = os.path.join(saved_params_path, dir)
    with open(os.path.join(saved_params_path, "saved_params_%d.npy" % iter), "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step=0.01, iterations=100000, anneal_every = 1000,
        print_every=10, useSaved=True, test_func = None, test_every=100,
        start_iterations=0, saved_params_file_name=None):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    anneal_every -- specifies number of iterations to modify step size
    print_every -- specifies how many iterations to output loss
    test_every -- specifies how many iterations to ouput accuracy

    Return:
    x -- the parameter value after SGD finishes
    """

    # get logger
    logger = Logger()
    if useSaved:
        start_iter, oldx, state = load_saved_params(saved_params_file_name, start_iterations)
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / anneal_every)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0
    # test init parama
    test_func(x)
    
    expcost = None

    for iter in range(start_iter, iterations + 1):

        if iter % SAVE_PARAMS_EVERY == 0 and iter != 0:
            save_params(iter, x, logger.get_dir())

        cost, grad = f(x)

        x -= step * grad

        if iter % print_every == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            logger.log("iter %d: expcost %f  -  cost %f" % (iter, expcost, cost))

        if iter % test_every == 0:
            # test the model
            test_func(x)

        if iter % anneal_every == 0:
            step *= 0.5


    return x


