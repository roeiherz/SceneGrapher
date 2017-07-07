from Module import Module
from data import *
from Utils.Logger import Logger

def eval(word_embed_size=50, visual_embed_size=2048):
    """
    Evaluate the module using TBD metrics
    :return:
    """
    filesmanager = FilesManager(overrides_filename="module_eval.yaml")
    logger = Logger()

    logger.log("Prepare Data")
    test_data, object_ids, predicate_ids = prepare_eval_data()


    # create Module
    logger.log("Create Module")
    module = Module(object_ids, predicate_ids, word_embed_size, visual_embed_size)
    #module.visual.initialize_networks(gpu_num=1, batch_num=1)

    logger.log("Load module weights")

    params = filesmanager.load_file("scene_graph_base_module.eval.module")
    module.set_weights(params)

    logger.log("Evaluate Predicates")
    module.predicate_class_recall(test_data, params, 5)

    logger.log("Evaluate")
    module.r_k_metric(test_data, 100, params)



if __name__ == "__main__":
    eval()

