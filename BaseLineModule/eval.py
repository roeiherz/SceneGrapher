import cPickle

from Module import Module
from data import *


def eval(model_filename="best_params_for_eval2.npy", word_embed_size=300, visual_embed_size=2048):
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
                    predicates_training_dir_name="Wed_Jun_14_20:25:16_2017")
    module.visual.initialize_networks(gpu_num=1, batch_num=1)

    print("Load module weights")
    with open(model_filename, "r") as f:
        params = cPickle.load(f)
        module.set_weights(params)

    print("Evaluate Predicates")
    module.predicate_class_recall(test_data, params, 5)

    print("Evaluate")
    module.r_k_metric(test_data, 100, params)



if __name__ == "__main__":
    eval()

