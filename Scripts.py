import timeit
import cPickle
import os
import numpy as np
from Data.VisualGenome.local import GetAllImageData, GetSceneGraph
from TrainCNN import VisualGenome_PICKLES_PATH
from keras_frcnn.Utils.Utils import create_folder, VG_PATCH_PATH, PREDICATES_COUNT_FILE, ENTITIES_FILE, \
    HIERARCHY_MAPPING, plot_graph
from keras_frcnn.Utils.data import create_mini_data_visual_genome, get_module_filter_data, get_filtered_data


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
        hierarchy_mapping_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + hierarchy_mapping_name),
                                      'wb')
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
                print("This is iteration number: {}".format(ind))
            # Updating index
            ind += 1

        except Exception as e:
            print("Problem with {0} in index: {1}".format(e, entity))

    save_pickles(classes_count=predicates_count, classes_count_name=PREDICATES_COUNT_FILE, iter='final')
    return 1


def process_objects(img_data, hierarchy_mapping):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param img_data: list of entities files
    :param hierarchy_mapping: dict of hierarchy_mapping
    :return: list of ObjectMapping
    """
    # Get the whole objects from entities
    objects_ind = 1
    correct_labels = hierarchy_mapping.keys()
    for img in img_data:

        # Get the objects per image
        objects = img.objects
        for object in objects:

            # Get the lable of object
            label = object.names[0]

            # Check if it is a correct label
            if label not in correct_labels:
                continue

            objects_ind += 1

    print("Number of objects are: {0}".format(objects_ind))


def graph_plot():
    """
    This function plot training and testing graph
    :return:
    """
    path_to_folder = "/home/roeih/SceneGrapher/Training/TrainingObjectsCNN/Sun_May_28_21:21:47_2017"
    path_to_folder = "/home/roeih/SceneGrapher/Training/TrainingPredicatesCNN/Sun_May_28_21:28:52_2017/"
    plot_graph(folder_path=path_to_folder)


def create_predicate_count(entities_file_read="", entities_file_save=""):
    """
    This function counts the number of instances of the predicates from entities
    :param entities_file_read: the file name (entities) which we will be read
    :param entities_file_save: the file name (entities) which we will be save
    :return: predict_count_dict - a dict with number of instances of predicates in entities
    """

    # Load entities
    entities_path_read = os.path.join(VisualGenome_PICKLES_PATH, entities_file_read)
    entities = cPickle.load(open(entities_path_read, "rb"))
    # Init dict
    predict_count_dict = {}

    for entity in entities:

        # Get relationships
        relations = entity.relationships
        for relation in relations:
            # Get the predicate
            predicate = relation.predicate

            # Update the classes_count dict
            if predicate in predict_count_dict:
                # Check if label is already in dict
                predict_count_dict[predicate] += 1
            else:
                # Init label in dict
                predict_count_dict[predicate] = 1

    # Pickle the predict count dict
    entities_path_save = os.path.join(VisualGenome_PICKLES_PATH, entities_file_save)
    f = open(entities_path_save, "wb")
    cPickle.dump(predict_count_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return predict_count_dict


def save_hierarchy_mapping():
    """
    This function save hierarchy mapping objects and hierarchy mapping predicates
    :return: 
    """

    # Get the hierarchy mapping
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "filtered_module_data.p")
    # Save hierarchy_mapping_per_objects file
    hierarchy_mapping_file = open(os.path.join(
        "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/VisualModule/Data/VisualGenome/hierarchy_mapping_objects.p"),
                                  'wb')
    # Pickle hierarchy_mapping_per_objects
    cPickle.dump(hierarchy_mapping_objects, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    # Save hierarchy_mapping_per_predicates file
    hierarchy_mapping_file2 = open(os.path.join(
        "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/VisualModule/Data/VisualGenome/hierarchy_mapping_predicates.p"),
                                   'wb')
    # Pickle hierarchy_mapping_per_objects
    cPickle.dump(hierarchy_mapping_predicates, hierarchy_mapping_file2, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file2.close()


if __name__ == '__main__':

    save_hierarchy_mapping()

    exit()
    # Filter the data
    filtered_module_data = get_module_filter_data(objects_count_file_name="mini_classes_count.p",
                                                  entities_file_name="final_entities.p",
                                                  predicates_count_file_name="mini_predicates_count.p", nof_objects=150,
                                                  nof_predicates=50)
    exit()

    # Create mini predicate count
    predict_count_dict = create_predicate_count(entities_file_read="mini_final_entities.p",
                                                entities_file_save="mini_predicates_count.p")

    # Check intersection of two hierarchies mapping
    hp_mini_path = "/home/roeih/SceneGrapher/Training/TrainingObjectsCNN/Sun_May_28_21:21:47_2017/class_mapping.p"
    hp_full_path = "/home/roeih/SceneGrapher/Training/TrainingObjectsCNN/Sat_May_27_18:25:10_2017_full/class_mapping.p"
    hierarchy_mapping_mini = cPickle.load(open(hp_mini_path, 'rb'))
    hierarchy_mapping_full = cPickle.load(open(hp_full_path, 'rb'))

    hierarchy_mapping_mini_set = set(hierarchy_mapping_mini.keys())
    hierarchy_mapping_full_set = set(hierarchy_mapping_full.keys())
    print("debug")

    # Graph plot
    graph_plot()

    # Create MINI data Visual Genome
    classes_count, hierarchy_mapping, entities = create_mini_data_visual_genome()
    entities_file_name = os.path.join(VisualGenome_PICKLES_PATH, ENTITIES_FILE)
    entities = cPickle.load(open(entities_file_name, 'rb'))

    hierarchy_mapping_file_name = os.path.join(VisualGenome_PICKLES_PATH, HIERARCHY_MAPPING)
    hierarchy_mapping = cPickle.load(open(hierarchy_mapping_file_name, 'rb'))
    process_objects(entities, hierarchy_mapping)
    # print(res)
