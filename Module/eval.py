from ModuleLogger import ModuleLogger
from WordEmbd import WordEmbd
from Module import Module
import sgd
import numpy as np
from data import *
from gradcheck import gradcheck_naive
import yaml
import inspect
import cPickle


def eval(model_filename="best_params_for_eval.npy", word_embed_size=300, visual_embed_size=2048):
    """
    Evaluate the module using TBD metrics
    :return:
    """
    print("Prepare Data")
    test_data, object_ids, predicate_ids = prepare_eval_data()


    # create Module
    print("Create Module")
    module = Module(object_ids, predicate_ids, word_embed_size, visual_embed_size,
                    objects_training_dir_name="Fri_Jun__2_19:16:26_2017",
                    predicates_training_dir_name="Fri_Jun__2_20:00:24_2017")
    module.visual.initialize_networks(gpu_num=2, batch_num=1)

    print("Load module weights")
    with open(model_filename, "r") as f:
        params = cPickle.load(f)
        module.set_weights(params)

    print("Evaluate")
    module.r_k_metric(test_data, 100, params)

if __name__ == "__main__":
    eval()

