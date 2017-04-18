from __future__ import print_function
from __future__ import print_function
import timeit
import cPickle
import os

from Data.VisualGenome.local import GetAllImageData, GetSceneGraph
from TrainCNN import VisualGenome_PICKLES_PATH, CLASSES_COUNT_FILE, CLASSES_MAPPING_FILE, ENTITIES_FILE, DATA_PATH, \
    RELATIONS_COUNT_FILE, RELATIONS_MAPPING_FILE, PREDICATES_COUNT_FILE
from keras_frcnn.Utils.Utils import create_folder, VG_PATCH_PATH


def check_loading_pickle_time():
    start = timeit.timeit()
    print("hello")
    f = cPickle.load(file("keras_frcnn/Data/VisualGenome/final_entities.p", 'rb'))

    end = timeit.timeit()
    print(end - start)


def save_pickles(classes_count=None, classes_count_name="", hierarchy_mapping=None, hierarchy_mapping_name="",
                 entities=None, iter=''):
    """
    This function save the pickles each iter
    :param hierarchy_mapping_name: hierarchy_mapping file name
    :param classes_count_name: classes_count file name
    :param classes_count: dict the classes count
    :param entities: dict with the entities
    :param hierarchy_mapping: dict with hierarchy mapping
    :param iter: string iteration number
    """

    # Check if classes_count is not None
    if classes_count:
        # Save classes_count file
        classes_count_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + classes_count_name), 'wb')
        # Pickle classes_count
        cPickle.dump(classes_count, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
        # Close the file
        classes_count_file.close()

    # Check if hierarchy_mapping is not None
    if hierarchy_mapping:
        # Save hierarchy_mapping file
        hierarchy_mapping_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + hierarchy_mapping_name), 'wb')
        # Pickle hierarchy_mapping
        cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
        # Close the file
        hierarchy_mapping_file.close()


def create_data():
    """
    This function creates hierarchy_relations and relations_counts
    :return:
    """

    predicates_count = {}

    # Map between relation to relation_id
    # relations_count = {}
    # relations_mapping = {}

    # load entities
    entities_file_name = os.path.join(VisualGenome_PICKLES_PATH, ENTITIES_FILE)
    entities = cPickle.load(file(entities_file_name, 'rb'))

    # index for saving
    ind = 1
    print("Start creating pickle for VisualGenome Data with changes")
    for entity in entities:
        try:
            relationships = entity.relationships
            for relation in relationships:
                predicate = relation.predicate

                # Update the classes_count dict
                if predicate in predicates_count:
                    # Check if label is already in dict
                    predicates_count[predicate] += 1
                else:
                    # Init label in dict
                    predicates_count[predicate] = 1

            # Printing Alerting
            if ind % 10000 == 0:
                save_pickles(classes_count=predicates_count, classes_count_name=PREDICATES_COUNT_FILE, iter=str(ind))
                # print("This is iteration number: {}".format(ind))
            # Updating index
            ind += 1
            print("This is iteration number: {}".format(entity))

        except Exception as e:
            print("Problem with {0} in index: {1}".format(e, entity))

    save_pickles(classes_count=predicates_count, classes_count_name=PREDICATES_COUNT_FILE, iter='final')
    return 1


if __name__ == '__main__':
    res = create_data()
    print(res)
