import traceback
import scipy.io
import matplotlib
from shutil import copyfile
from keras.engine import Input, Model
from keras.layers import GlobalAveragePooling2D, Dense
from Data.VisualGenome.models import Object, Graph, Image, Relationship, ObjectMapping, RelationshipMapping, \
    ImageMapping
from FeaturesExtraction.Lib.Zoo import ModelZoo
from FeaturesExtraction.Utils.Boxes import find_union_box
from FeaturesExtraction.Utils.Visualizer import VisualizerDrawer, CvColor
from FeaturesExtraction.Utils.VisualizerEntity import VisualizerEntity

# matplotlib.use('agg')
from matplotlib import pyplot as plt
import time
import cPickle
import os
import numpy as np
from Data.VisualGenome.local import GetAllImageData, GetSceneGraph, GetAllRegionDescriptions
from FilesManager.FilesManager import FilesManager
from FeaturesExtraction.Utils.Utils import VG_PATCH_PATH, PREDICATES_COUNT_FILE, ENTITIES_FILE, \
    HIERARCHY_MAPPING, plot_graph, POSITIVE_NEGATIVE_RATIO, DATA_PATH, CLASSES_COUNT_FILE, RELATIONS_COUNT_FILE, \
    VisualGenome_PICKLES_PATH, TRAINING_OBJECTS_CNN_PATH, WEIGHTS_NAME, TRAINING_PREDICATE_CNN_PATH, \
    PREDICATED_FEATURES_PATH, get_time_and_date, get_img, FILTERED_DATA_SPLIT_PATH, PROJECT_ROOT, \
    EXTRACTED_DATA_SPLIT_PATH, DATA, VISUAL_GENOME, VG_DATA_PATH, PREPROCESSED_FILTERED_DATA_SPLIT_PATH
from TrainCNN import preprocessing_objects
from Utils.Utils import create_folder
from FeaturesExtraction.Utils.data import create_mini_data_visual_genome, get_module_filter_data, get_filtered_data, \
    visualize_detections, process_to_detections
from PredictVisualModel import get_resize_images_array, load_full_detections, get_model
from FeaturesExtraction.Utils.Utils import VG_VisualModule_PICKLES_PATH
from FeaturesExtraction.Lib.Config import Config
from DesignPatterns.Detections import Detections
from Utils.Logger import Logger
from keras import backend as K
import pandas as pd
import tensorflow as tf
import cv2


def check_loading_pickle_time():
    start = time.time()
    print("hello")
    f = cPickle.load(open("keras_frcnn/Data/VisualGenome/final_entities.p"))

    end = time.time()
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
        hierarchy_mapping_file = open(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + hierarchy_mapping_name),
                                      'wb')
        # Pickle hierarchy_mapping
        cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
        # Close the file
        hierarchy_mapping_file.close()


def create_data_object_and_predicates_by_img_id():
    """
    This function creates predicates dict and objects dict with number if instances
    :return:
    """
    img_ids_lst = get_img_ids()
    ind = 1
    objects_count = {}
    predicates_count = {}
    entities_lst = []

    for img_id in img_ids_lst:
        try:
            entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
                                   synsetFile=DATA_PATH + "synsets.json")
            entities_lst.append(entity)
            objects = entity.objects
            for object in objects:
                label = object.names[0]

                # Update the classes_count dict
                if label in objects_count:
                    # Check if label is already in dict
                    objects_count[label] += 1
                else:
                    # Init label in dict
                    objects_count[label] = 1

            predicates = entity.relationships
            for relation in predicates:
                predicate = relation.predicate

                # Update the classes_count dict
                if predicate in predicates_count:
                    # Check if label is already in dict
                    predicates_count[predicate] += 1
                else:
                    # Init label in dict
                    predicates_count[predicate] = 1

            # Updating index
            ind += 1
            print("This is iteration number: {}".format(img_id))

        except Exception as e:
            print("Problem with {0} in index: {1}".format(e, img_id))

    # Save objects_count file
    classes_count_file = file(CLASSES_COUNT_FILE, 'wb')
    # Pickle classes_count
    cPickle.dump(objects_count, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()

    # Save classes_count file
    relations_count_file = file(RELATIONS_COUNT_FILE, 'wb')
    # Pickle classes_count
    cPickle.dump(predicates_count, relations_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    relations_count_file.close()

    # Save entities list
    entities_file = file(ENTITIES_FILE, 'wb')
    # Pickle entities
    cPickle.dump(entities_lst, entities_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    entities_file.close()


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
    path_to_folder = "/home/roeih/SceneGrapher/FilesManager/FeaturesExtraction/PredicatesMaskCNN/Best_so_far_121117/Fri_Oct_27_22:41:05_2017"
    plot_graph(folder_path=path_to_folder)


def create_predicate_count(entities_file_read="", entities_file_save=""):
    """
    This function counts the number of instances of the predicates from entities
    :param entities_file_read: the file name (entities) which we will be read
    :param entities_file_save: the file name (entities) which we will be save
    :return: predict_count_dict - a dict with number of instances of predicates in entities
    """

    # Load entities
    # entities_path_read = os.path.join(VisualGenome_PICKLES_PATH, entities_file_read)
    # entities = cPickle.load(open(entities_path_read, "rb"))
    # Load detections dtype numpy array and hierarchy mappings
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "full_filtered_data",
                                                                                          category='entities')

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


def save_union_detections(path="mini_resize_union_detections.p"):
    """
    This function save the union detections
    """
    config = Config(1)

    detections = load_full_detections(detections_file_name="mini_filtered_detections.p")

    detections = detections[:500]

    print("Saving union predicated_detections")
    resized_img_mat = get_resize_images_array(detections, config)
    print("Saving predicated_detections")
    # Save detections
    detections_filename = open(os.path.join(VG_VisualModule_PICKLES_PATH, path), 'wb')
    # Pickle detections
    cPickle.dump(resized_img_mat, detections_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    detections_filename.close()
    print("Finished successfully saving predicated_detections")


def delete_ind_from_detections():
    """
    This function removes detections with specific indices
    """
    detections = load_full_detections(detections_file_name="mini_filtered_detections.p")

    idx = np.where((detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K/2321818.jpg") |
                   (detections[Detections.Url] == "https://cs.stanford.edu/people/rak248/VG_100K/2334844.jpg"))
    new_detections = np.delete(detections, idx)

    # Save detections
    detections_path = os.path.join(VG_VisualModule_PICKLES_PATH, "mini_fixed_filtered_detections.p")
    detections_filename = open(detections_path, 'wb')
    # Pickle detections
    cPickle.dump(new_detections, detections_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    detections_filename.close()


def get_img_ids():
    """
    This function returns the 1000 image_ids by FEI-FEI paper
    :return:
    """
    img_ids = [2339172, 2339171, 2339170, 2339169, 2339168, 2339167, 2339166,
               2339165, 2339164, 2339163, 2339162, 2339161, 2339160, 2339159,
               2339158, 2339157, 2339156, 2339155, 2339154, 2339153, 2339152,
               2339151, 2339150, 2339149, 2339148, 2339147, 2339146, 2339145,
               2339144, 2339143, 2339142, 2339141, 2339140, 2339139, 2339138,
               2339137, 2339136, 2339135, 2339134, 2339133, 2339132, 2339131,
               2339130, 2339129, 2339127, 2339126, 2339125, 2339124, 2339123,
               2339122, 2339121, 2339120, 2339119, 2339118, 2339117, 2339116,
               2339115, 2339114, 2339113, 2339112, 2339110, 2339109, 2339108,
               2339107, 2339106, 2339105, 2339104, 2339103, 2339102, 2339101,
               2339100, 2339099, 2339098, 2339097, 2339096, 2339095, 2339094,
               2339093, 2339092, 2339091, 2339090, 2339089, 2339088, 2339087,
               2339086, 2339085, 2339084, 2339083, 2339082, 2339081, 2339080,
               2339079, 2339078, 2339077, 2339076, 2339075, 2339074, 2339073,
               2339072, 2339071, 2339070, 2339069, 2339068, 2339067, 2339066,
               2339065, 2339064, 2339063, 2339062, 2339061, 2339060, 2339059,
               2339058, 2339057, 2339056, 2339055, 2339054, 2339053, 2339052,
               2339051, 2339050, 2339049, 2339048, 2339047, 2339046, 2339045,
               2339044, 2339043, 2339042, 2339041, 2339040, 2339039, 2339037,
               2339036, 2339034, 2339033, 2339031, 2339030, 2339029, 2339028,
               2339027, 2339026, 2339025, 2339024, 2339023, 2339022, 2339021,
               2339020, 2339019, 2339018, 2339017, 2339016, 2339015, 2339014,
               2339013, 2339012, 2339011, 2339010, 2339009, 2339008, 2339007,
               2339006, 2339005, 2339004, 2339003, 2339002, 2339001, 2339000,
               2338999, 2338998, 2338997, 2338996, 2338995, 2338994, 2338993,
               2338992, 2338991, 2338990, 2338989, 2338988, 2338987, 2338986,
               2338985, 2338984, 2338983, 2338982, 2338981, 2338980, 2338979,
               2338978, 2338977, 2338976, 2338975, 2338974, 2338973, 2338972,
               2338971, 2338970, 2338969, 2338968, 2338967, 2338966, 2338965,
               2338964, 2338963, 2338962, 2338961, 2338960, 2338959, 2338958,
               2338957, 2338956, 2338955, 2338954, 2338953, 2338952, 2338951,
               2338950, 2338949, 2338948, 2338947, 2338946, 2338945, 2338944,
               2338943, 2338942, 2338941, 2338940, 2338939, 2338938, 2338937,
               2338936, 2338935, 2338934, 2338933, 2338932, 2338931, 2338930,
               2338929, 2338928, 2338927, 2338925, 2338924, 2338923, 2338922,
               2338921, 2338920, 2338919, 2338918, 2338917, 2338916, 2338915,
               2338914, 2338913, 2338912, 2338911, 2338910, 2338909, 2338908,
               2338907, 2338906, 2338905, 2338904, 2338903, 2338902, 2338901,
               2338900, 2338899, 2338898, 2338897, 2338896, 2338895, 2338894,
               2338893, 2338891, 2338890, 2338889, 2338888, 2338887, 2338886,
               2338885, 2338882, 2338881, 2338880, 2338879, 2338878, 2338877,
               2338876, 2338875, 2338874, 2338873, 2338872, 2338871, 2338870,
               2338869, 2338868, 2338867, 2338866, 2338865, 2338864, 2338863,
               2338862, 2338861, 2338860, 2338859, 2338858, 2338857, 2338856,
               2338855, 2338854, 2338853, 2338852, 2338851, 2338850, 2338849,
               2338848, 2338847, 2338846, 2338845, 2338844, 2338843, 2338842,
               2338841, 2338840, 2338839, 2338838, 2338837, 2338836, 2338835,
               2338834, 2338833, 2338831, 2338830, 2338829, 2338828, 2338827,
               2338826, 2338825, 2338824, 2338822, 2338821, 2338820, 2338819,
               2338817, 2338816, 2338815, 2338814, 2338812, 2338811, 2338810,
               2338809, 2338808, 2338807, 2338806, 2338805, 2338804, 2338803,
               2338802, 2338801, 2338799, 2338798, 2338797, 2338796, 2338795,
               2338794, 2338793, 2338792, 2338790, 2338789, 2338788, 2338787,
               2338786, 2338785, 2338784, 2338783, 2338782, 2338781, 2338780,
               2338779, 2338778, 2338777, 2338776, 2338775, 2338774, 2338773,
               2338772, 2338771, 2338770, 2338768, 2338767, 2338766, 2338765,
               2338763, 2338762, 2338761, 2338760, 2338759, 2338757, 2338756,
               2338755, 2338754, 2338753, 2338752, 2338751, 2338750, 2338749,
               2338748, 2338746, 2338745, 2338744, 2338743, 2338742, 2338740,
               2338739, 2338738, 2338737, 2338736, 2338735, 2338734, 2338733,
               2338732, 2338731, 2338730, 2338729, 2338728, 2338727, 2338726,
               2338725, 2338724, 2338723, 2338722, 2338721, 2338720, 2338719,
               2338718, 2338717, 2338716, 2338715, 2338714, 2338713, 2338712,
               2338711, 2338710, 2338709, 2338708, 2338707, 2338706, 2338705,
               2338704, 2338703, 2338702, 2338701, 2338700, 2338699, 2338698,
               2338697, 2338696, 2338695, 2338694, 2338693, 2338692, 2338691,
               2338690, 2338689, 2338688, 2338687, 2338686, 2338685, 2338684,
               2338683, 2338682, 2338681, 2338680, 2338679, 2338678, 2338677,
               2338676, 2338675, 2338674, 2338673, 2338672, 2338671, 2338670,
               2338669, 2338668, 2338666, 2338665, 2338664, 2338663, 2338662,
               2338661, 2338660, 2338659, 2338658, 2338657, 2338656, 2338655,
               2338653, 2338652, 2338651, 2338650, 2338649, 2338648, 2338647,
               2338646, 2338645, 2338644, 2338643, 2338642, 2338641, 2338640,
               2338639, 2338638, 2338637, 2338636, 2338634, 2338633, 2338632,
               2338631, 2338630, 2338629, 2338628, 2338627, 2338626, 2338625,
               2338624, 2338621, 2338620, 2338619, 2338618, 2338617, 2338616,
               2338615, 2338614, 2338613, 2338612, 2338611, 2338610, 2338609,
               2338608, 2338607, 2338606, 2338605, 2338603, 2338602, 2338601,
               2338600, 2338599, 2338598, 2338597, 2338596, 2338595, 2338594,
               2338593, 2338592, 2338591, 2338590, 2338589, 2338588, 2338587,
               2338586, 2338585, 2338584, 2338583, 2338582, 2338581, 2338580,
               2338578, 2338577, 2338575, 2338574, 2338573, 2338572, 2338571,
               2338570, 2338569, 2338568, 2338567, 2338566, 2338565, 2338564,
               2338563, 2338562, 2338561, 2338560, 2338559, 2338558, 2338557,
               2338556, 2338555, 2338554, 2338553, 2338552, 2338551, 2338550,
               2338549, 2338548, 2338547, 2338546, 2338545, 2338544, 2338543,
               2338542, 2338541, 2338540, 2338539, 2338538, 2338537, 2338536,
               2338535, 2338534, 2338533, 2338532, 2338531, 2338530, 2338529,
               2338528, 2338527, 2338526, 2338525, 2338524, 2338523, 2338522,
               2338521, 2338520, 2338519, 2338518, 2338517, 2338516, 2338515,
               2338514, 2338513, 2338512, 2338511, 2338510, 2338509, 2338508,
               2338507, 2338506, 2338505, 2338504, 2338503, 2338502, 2338501,
               2338500, 2338499, 2338498, 2338496, 2338495, 2338494, 2338493,
               2338492, 2338491, 2338490, 2338488, 2338487, 2338486, 2338485,
               2338484, 2338483, 2338482, 2338481, 2338480, 2338479, 2338478,
               2338477, 2338476, 2338475, 2338474, 2338473, 2338472, 2338471,
               2338470, 2338469, 2338468, 2338467, 2338466, 2338465, 2338464,
               2338463, 2338462, 2338461, 2338460, 2338459, 2338458, 2338457,
               2338456, 2338455, 2338454, 2338453, 2338452, 2338451, 2338450,
               2338449, 2338448, 2338447, 2338446, 2338445, 2338444, 2338443,
               2338442, 2338441, 2338440, 2338439, 2338438, 2338437, 2338436,
               2338435, 2338434, 2338433, 2338432, 2338431, 2338430, 2338429,
               2338428, 2338427, 2338425, 2338424, 2338423, 2338422, 2338421,
               2338420, 2338419, 2338418, 2338417, 2338416, 2338415, 2338414,
               2338413, 2338412, 2338411, 2338410, 2338409, 2338408, 2338407,
               2338406, 2338405, 2338404, 2338402, 2338401, 2338400, 2338399,
               2338398, 2338396, 2338395, 2338394, 2338393, 2338392, 2338391,
               2338390, 2338389, 2338388, 2338386, 2338385, 2338384, 2338383,
               2338382, 2338381, 2338380, 2338379, 2338378, 2338377, 2338376,
               2338375, 2338374, 2338373, 2338372, 2338371, 2338370, 2338369,
               2338368, 2338367, 2338366, 2338365, 2338364, 2338363, 2338362,
               2338361, 2338360, 2338359, 2338357, 2338356, 2338355, 2338354,
               2338353, 2338350, 2338349, 2338348, 2338347, 2338346, 2338345,
               2338344, 2338343, 2338342, 2338341, 2338340, 2338339, 2338338,
               2338337, 2338336, 2338335, 2338334, 2338333, 2338332, 2338331,
               2338330, 2338329, 2338328, 2338326, 2338325, 2338324, 2338323,
               2338322, 2338321, 2338320, 2338319, 2338318, 2338317, 2338316,
               2338315, 2338314, 2338313, 2338312, 2338311, 2338310, 2338309,
               2338308, 2338307, 2338306, 2338305, 2338304, 2338303, 2338302,
               2338301, 2338300, 2338299, 2338298, 2338297, 2338296, 2338295,
               2338294, 2338293, 2338292, 2338291, 2338290, 2338289, 2338288,
               2338287, 2338286, 2338285, 2338284, 2338283, 2338282, 2338281,
               2338280, 2338279, 2338278, 2338277, 2338276, 2338275, 2338274,
               2338273, 2338272, 2338271, 2338270, 2338269, 2338268, 2338267,
               2338266, 2338265, 2338264, 2338263, 2338262, 2338261, 2338260,
               2338259, 2338258, 2338257, 2338256, 2338255, 2338254, 2338253,
               2338252, 2338251, 2338250, 2338249, 2338247, 2338246, 2338245,
               2338244, 2338242, 2338241, 2338240, 2338239, 2338238, 2338237,
               2338236, 2338235, 2338234, 2338233, 2338232, 2338231, 2338230,
               2338229, 2338228, 2338227, 2338226, 2338225, 2338224, 2338223,
               2338222, 2338221, 2338220, 2338219, 2338218, 2338217, 2338216,
               2338215, 2338214, 2338213, 2338212, 2338211, 2338210, 2338209,
               2338208, 2338207, 2338206, 2338205, 2338204, 2338202, 2338201,
               2338200, 2338199, 2338198, 2338197, 2338196, 2338195, 2338194,
               2338193, 2338192, 2338191, 2338190, 2338189, 2338188, 2338187,
               2338186, 2338185, 2338184, 2338183, 2338181, 2338180, 2338178,
               2338177, 2338176, 2338175, 2338174, 2338173, 2338172, 2338171,
               2338170, 2338169, 2338168, 2338167, 2338166, 2338165, 2338164,
               2338163, 2338162, 2338161, 2338160, 2338159, 2338158, 2338157,
               2338156, 2338155, 2338154, 2338153, 2338152, 2338151, 2338150,
               2338149, 2338148, 2338147, 2338146, 2338145, 2338144, 2338143,
               2338142, 2338141, 2338140, 2338139, 2338138, 2338137, 2338136,
               2338135, 2338134, 2338133, 2338131, 2338130, 2338129]

    return img_ids


def get_mini_url():
    """
    This function saves in list the urls from mini dataset
    :return: 
    """

    # Load entities
    # entities, _, _ = get_filtered_data("filtered_module_data_with_neg.p", category='entities')
    entities, _, _ = get_filtered_data("final_filtered_module_data_with_neg.p", category='entities')

    # Get Url list from entities
    url_lst = [entity.image.url for entity in entities]

    # Save mini_filtered_module_data url's list
    # url_file = open(os.path.join(VisualGenome_PICKLES_PATH, "full_url_lst_mini.p"), 'wb')
    url_file = open(os.path.join(VisualGenome_PICKLES_PATH, "final_full_url_lst_mini.p"), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(url_lst, url_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    url_file.close()

    exit()
    # Load detections
    detections_path = os.path.join(VG_VisualModule_PICKLES_PATH, "predicated_mini_fixed_detections.p")
    detections = cPickle.load(open(detections_path, 'rb'))

    # Filtered detections
    indx = np.where(np.in1d(list(detections[Detections.Url]), url_lst) == True)
    filtered_detections = detections[indx]

    # Save filtered detections
    filtered_detections_file = open(os.path.join(VisualGenome_PICKLES_PATH, "mini_filtered_module_data_url.p"), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(filtered_detections, filtered_detections_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    filtered_detections_file.close()
    print("debug")


def Objects_AR_Histogram():
    """
    This functions calcs the Aspect Ration in Objects in VisualGenome dataset
    """

    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(
        filtered_data_file_name="full_filtered_data")
    # Get Visual Genome Data objects
    objects = preprocessing_objects(entities, hierarchy_mapping_objects, object_file_name="full_objects")
    objects_ar_list = [object.height / float(object.width) for object in objects]
    plt.figure()
    plt.hist(objects_ar_list, bins=100, range=[0, 20], normed=1, histtype='bar')
    plt.title('Objects Aspect Ratio histogram')
    plt.savefig("Objects_AR_Histogram.jpg")


def test_predicted_features(gpu_num=0, objects_training_dir_name="", predicates_training_dir_name="",
                            predicted_features_dir_name="", predicted_features_file_name=""):
    """
    This function loads the predicted features from PredictFeaturesModule output, and checks the features from
        objects and predicates are fit to the probabilities
    :param gpu_num: number of gpu which will be used
    :param objects_training_dir_name: the name of the folder which we will be loading object model
    :param predicates_training_dir_name: the name of the folder which we will be loading predicate model
    :param predicted_features_dir_name: the name of the folder which we will be loading predicted features
    :param predicted_features_file_name: the name of the file which we will be loading predicted features
    :return: 
    """

    # Load class config
    config = Config(gpu_num)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)

    # Load detections dtype numpy array and hierarchy mappings
    _, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                   "mini_filtered_data",
                                                                                   category='entities_visual_module')

    # Load the weight paths
    objects_model_weight_path = os.path.join(TRAINING_OBJECTS_CNN_PATH, objects_training_dir_name,
                                             WEIGHTS_NAME)
    predicates_model_weight_path = os.path.join(TRAINING_PREDICATE_CNN_PATH, predicates_training_dir_name,
                                                WEIGHTS_NAME)
    predicted_features_path = os.path.join(PREDICATED_FEATURES_PATH, predicted_features_dir_name)

    predicted_features_file_name = os.path.join(predicted_features_path, predicted_features_file_name)

    Logger().log("Loading the predicted features file")
    predicted_features_entities = cPickle.load(open(predicted_features_file_name))

    # Set the number of classes
    number_of_classes_objects = len(hierarchy_mapping_objects)
    number_of_classes_predicates = len(hierarchy_mapping_predicates)

    Logger().log("Loading the object and predicate models")
    object_model = get_model(number_of_classes_objects, weight_path=objects_model_weight_path, config=config)
    predicate_model = get_model(number_of_classes_predicates, weight_path=predicates_model_weight_path, config=config)

    features_output_object_func = K.function([object_model.layers[-1].input],
                                             [object_model.layers[-1].output])

    features_output_predicate_func = K.function([predicate_model.layers[-1].input],
                                                [predicate_model.layers[-1].output])
    for entity in predicted_features_entities:
        try:
            Logger().log("Start testing Entity: {0}".format(entity.image.id))
            # Assert objects probabilites
            objects_probes = features_output_object_func([entity.objects_features])[0]
            np.testing.assert_almost_equal(entity.objects_probs, objects_probes, decimal=5)

            # Assert predicates probabilites
            # [len(objects) * len(objects), 2048]
            reshaped_predicate_features = entity.predicates_features.reshape(len(entity.objects) * len(entity.objects),
                                                                             2048)
            predicates_probes = features_output_predicate_func([reshaped_predicate_features])[0]

            # [len(objects), len(objects), 51]
            predicates_probes_reshaped = predicates_probes.reshape(len(entity.objects), len(entity.objects), 51)
            np.testing.assert_almost_equal(entity.predicates_probes, predicates_probes_reshaped, decimal=5)
        except Exception as e:
            Logger().log('Exception in image_id: {0} with error: {1}'.format(entity.image.id, e))
            Logger().log(str(e))
            traceback.print_exc()

    Logger().log("Finished testing")


def get_bad_img_ids_from_logger():
    """
    This function returns bad image ids that have ** Error in the logger from the run
    :return:
    """
    files = ["PredictFeaturesModule_0_to_18013.log", "PredictFeaturesModule_18013_to_36026.log"]
    bad_img_ids = set([])
    for fl in files:
        with open(fl) as f:
            for line in f:
                if "**" in line:
                    bad_img_id = int(line.split(" ")[4])
                    bad_img_ids.add(bad_img_id)

    return bad_img_ids


def logger_parser(dir_path="Temp"):
    """
    This function returns good image ids from logger from the original run
    :return:
    """

    # Define from each files we are gonna to parse
    files = ["PredictFeaturesModule_0_to_18013.log", "PredictFeaturesModule_18013_to_36026.log"]
    # files = ["PredictFeaturesModule_mini_entities.log"]
    files = ["PredictFeaturesModule_mini_entities_module_mask.log"]
    files = ["PredictFeaturesModule_mini_entities_mask_dual_260917.log"]
    files = ["PredictFeaturesModule_mini_entities_updated_231017.log"]
    files = ["PredictFeaturesModule_train_split0_old_211017.log"]
    files = ["PredictFeaturesModule_test_split0_251017.log"]
    files = ["PredictFeaturesModule_test_mini_objects_july_011117.log"]
    files = ["PredictFeaturesModule_test_mini_objects_imgnet_011117.log"]
    files = ["PredictFeaturesModule_test_split0_011117.log"]
    files = ["PredictFeaturesModule_test_split0_july_031117.log"]
    files = ["PredictFeaturesModule_minientities_imgnet_workers_cutoff.log"]

    # Get the data frame from logger
    df = create_dataframe_from_logger(files)

    # Get time and date
    time_and_date = get_time_and_date()
    dir_path = os.path.join(dir_path, time_and_date)
    create_folder(dir_path)

    # Save DataFrame
    df.to_csv(os.path.join(dir_path, "test_minientities_imgnet_workers_cutoff_111117.csv"))
    fl = open(os.path.join(dir_path, "test_minientities_imgnet_workers_cutoff_111117_df.p"), "wb")
    cPickle.dump(df, fl)
    fl.close()


def create_dataframe_from_logger(files):
    """
    This function will create data frame from files which are loggers
    :param files: a list of files which are the logger
    :return:
    """

    # Define the rows for the DataFrame
    dataframe_labels = ["Image_Id", "Total_Objects", "Number_Of_Positive_Objects", "Number_Of_Negative_Objects",
                        "Objects_Accuracy", "Total_Relations", "Number_Of_Positive_Relations",
                        "Number_Of_Negative_Relations", "Relations_Accuracy", "Positive_Relations_Accuracy",
                        "Negative_Relations_Accuracy", "Error"]

    # Define DataFrame
    df = pd.DataFrame(columns=dataframe_labels)

    for fl in files:
        with open(fl) as f:
            for line in f:
                # Check if we a new section (each "Predicting image id" is a new section to read)
                if "Predicting image id" in line:
                    row_data = {}
                    image_id = int(line.split(" ")[3])
                    row_data["Image_Id"] = image_id
                    row_data["Error"] = False

                if "The Total number of Objects" in line:
                    total_objects = int(line.split(" ")[6])
                    positive_objects = int(line.split(" ")[8])
                    row_data["Total_Objects"] = total_objects
                    row_data["Number_Of_Positive_Objects"] = positive_objects
                    row_data["Number_Of_Negative_Objects"] = total_objects - positive_objects

                if "The Objects accuracy" in line:
                    object_accuracy = float(line.split(" ")[-1])
                    row_data["Objects_Accuracy"] = object_accuracy

                # Found an error
                if "**" in line:
                    row_data["Error"] = True

                if "The Total number of Relations" in line:
                    total_relations = int(line.split(" ")[6])
                    positive_relations = int(line.split(" ")[8])
                    negative_relations = int(line.split(" ")[-5])
                    row_data["Total_Relations"] = total_relations
                    row_data["Number_Of_Positive_Relations"] = positive_relations
                    row_data["Number_Of_Negative_Relations"] = negative_relations

                if "The Total Relations accuracy" in line:
                    relation_accuracy = float(line.split(" ")[-1])
                    row_data["Relations_Accuracy"] = relation_accuracy

                if "The Positive Relations accuracy" in line:
                    positive_relation_accuracy = float(line.split(" ")[5])
                    row_data["Positive_Relations_Accuracy"] = positive_relation_accuracy

                if "The Negative Relations accuracy" in line:
                    negative_relation_accuracy = float(line.split(" ")[-1])
                    row_data["Negative_Relations_Accuracy"] = negative_relation_accuracy

                    # Section ends in this line
                    # Adding a row
                    df.loc[-1] = row_data
                    # Shifting index
                    df.index = df.index + 1
                    # Sorting by index
                    df = df.sort()

    return df


def detection_parser(dir_path="Temp", files=[]):
    """
    This function returns csv file from detection pickle file (Detections numpy array)
    :return:
    """

    # Define from each files we are gonna to parse
    if len(files) == 0:
        files = ["mini_predicated_mask_predicates_Mon_Sep_25_17:47:17_2017_260917.p"]

    # Get the data frame from logger
    df = create_dataframe_from_detection(files)

    # Get time and date
    time_and_date = get_time_and_date()
    dir_path = os.path.join(dir_path, time_and_date)
    create_folder(dir_path)

    # Make queries
    # Calc the accuracy of the Predicates
    predicates_nof_eq = df[df.predicate == df.predict_predicate].groupby(df.predicate).count()
    total_nof_predicates = df.groupby(df.predicate).count()
    predicates_acc = (predicates_nof_eq / total_nof_predicates)["id"]
    # Save the accuracy of the Predicates
    predicates_acc.to_csv(os.path.join(dir_path, "predicting_predicates_accuracy.csv"))

    # Calc the accuracy of the Subjects
    subjects_nof_eq = df[df.subject_classifications == df.predict_subject_classifications].groupby(
        df.subject_classifications).count()
    total_nof_subjects = df.groupby(df.subject_classifications).count()
    subjects_acc = (subjects_nof_eq / total_nof_subjects)["id"]
    # Save the accuracy of the Subjects
    subjects_acc.to_csv(os.path.join(dir_path, "predicting_subjects_accuracy.csv"))

    # Calc the accuracy of the Objects
    objects_nof_eq = df[df.object_classifications == df.predict_object_classifications].groupby(
        df.object_classifications).count()
    total_nof_objects = df.groupby(df.object_classifications).count()
    objects_acc = (objects_nof_eq / total_nof_objects)["id"]
    # Save the accuracy of the Objects
    objects_acc.to_csv(os.path.join(dir_path, "predicting_objects_accuracy.csv"))

    # Save DataFrame
    df.to_csv(os.path.join(dir_path, "logger_data_detections_Mon_Sep_25_17:47:17_2017_260917.csv"))
    fl = open(os.path.join(dir_path, "logger_data_detections_Mon_Sep_25_17:47:17_2017_260917_df.p"), "wb")
    cPickle.dump(df, fl)
    fl.close()


def create_dataframe_from_detection(files):
    """
    This function will create data frame from detections which are Detections numpy array
    :param files: a list of files which are the logger
    :return:
    """

    # Define the rows for the DataFrame
    dataframe_labels = [Detections.Id, Detections.SubjectBox, Detections.SubjectId, Detections.ObjectBox,
                        Detections.ObjectId, Detections.Predicate, "predict_predicate", Detections.UnionBox,
                        Detections.SubjectClassifications, Detections.PredictSubjectClassifications,
                        Detections.ObjectClassifications, Detections.PredictObjectClassifications,
                        Detections.SubjectConfidence, Detections.ObjectConfidence, Detections.Url]

    # Define DataFrame
    df = pd.DataFrame(columns=dataframe_labels)

    for fl in files:
        with open(fl) as f:
            detection_file = cPickle.load(f)
            for detection in detection_file:
                row_data = dict()
                row_data[Detections.Id] = int(detection[Detections.Id])
                row_data[Detections.SubjectBox] = detection[Detections.SubjectBox]
                row_data[Detections.SubjectId] = int(detection[Detections.SubjectId])
                row_data[Detections.ObjectBox] = detection[Detections.ObjectBox]
                row_data[Detections.ObjectId] = int(detection[Detections.ObjectId])
                row_data[Detections.Predicate] = detection[Detections.Predicate]
                row_data[Detections.UnionBox] = detection[Detections.UnionBox]
                row_data[Detections.SubjectClassifications] = detection[Detections.SubjectClassifications]
                row_data[Detections.PredictSubjectClassifications] = detection[Detections.PredictSubjectClassifications]
                row_data[Detections.ObjectClassifications] = detection[Detections.ObjectClassifications]
                row_data[Detections.PredictObjectClassifications] = detection[Detections.PredictObjectClassifications]
                row_data["predict_predicate"] = detection[Detections.UnionFeature]

                # Probabilities are not interesting
                row_data[Detections.SubjectConfidence] = 0
                row_data[Detections.ObjectConfidence] = 0

                row_data[Detections.Url] = detection[Detections.Url]

                # Section ends in this line
                # Adding a row
                df.loc[-1] = row_data
                # Shifting index
                df.index = df.index + 1
                # Sorting by index
                df = df.sort()

    return df


def get_predicates_dict_from_entities():
    global entities
    entities, hierarchy_mapping_objects, _ = get_filtered_data(filtered_data_file_name="mini_filtered_data",
                                                               category="entities")
    predicates_dict = {}
    for entity in entities:
        predicates = entity.relationships
        for pred in predicates:
            rel = pred.predicate

            if rel in predicates_dict:
                predicates_dict[rel] += 1
            else:
                predicates_dict[rel] = 1


def find_image_id_via_object(object=""):
    """
    This function find the image id by same object
    :param object: the object category that we are searching
    :return:
    """
    # Filter the data
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "full_filtered_data",
                                                                                          category='entities')
    for entity in entities:
        for obj in entity.objects:
            if obj.names[0] == object:
                # Get image - the pair is from the same image
                img = get_img(entity.image.url, download=True)
                cv2.imwrite(
                    "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Temp/Elephant/{}.jpg".format(entity.image.id), img)
                break


def split_filter_data():
    """
    This function splits the filter data
    :return:
    """
    # Filter the data
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "full_filtered_data",
                                                                                          category='entities')

    path = os.path.join(PROJECT_ROOT, FILTERED_DATA_SPLIT_PATH)
    create_folder(path)
    for entity in entities:
        path_fl = os.path.join(path, "{0}.p".format(str(entity.image.id)))
        with open(path_fl, "wb") as fl:
            cPickle.dump(entity, fl)
        print("Finish to save {0}".format(entity.image.id))


def split_extracted_data():
    """
    This function splits the extracted data (First Module Feature-Extraction Module)
    :return:
    """

    # Folders to be extracted
    folders = ["Wed_Sep_27_20:58:45_2017", "Fri_Sep_29_14:11:13_2017", "Sat_Sep_30_12:37:31_2017",
               "Thu_Oct__5_12:02:32_2017", "Thu_Oct__5_21:55:07_2017", "Mon_Oct__9_12:49:00_2017"]
    folders = ["Tue_Oct__3_22:13:00_2017"]

    # Get Paths
    save_path = os.path.join(PROJECT_ROOT, EXTRACTED_DATA_SPLIT_PATH)
    folders_path = os.path.join(PROJECT_ROOT, PREDICATED_FEATURES_PATH)
    create_folder(save_path)

    for folder in folders:
        dir_path = os.path.join(folders_path, folder)
        dir_files = os.listdir(dir_path)
        for pickle_file in dir_files:

            # Check the file is not a logger
            if ".log" in pickle_file:
                continue

            pickle_file_path = os.path.join(dir_path, pickle_file)
            entities = cPickle.load(open(pickle_file_path))
            for entity in entities:
                entity_path = os.path.join(save_path, "{0}.p".format(entity.image.id))
                with open(entity_path, "wb") as fl:
                    cPickle.dump(entity, fl)
                print("Finish to save {0}".format(entity.image.id))


def load_h5py():
    import h5py
    filename = '/home/roeih/VisualGenome/full/VG-SGG.h5'
    f = h5py.File(filename, 'r')
    # List all groups
    print("Keys: %s" % f.keys())
    labels = f['labels']
    split = f['split']
    from Data.VisualGenome.local import GetAllImageData, GetSceneGraph
    import os
    images = {img.id: img for img in GetAllImageData("/home/roeih/SceneGrapher/Data/VisualGenome/data/")}
    tt = GetSceneGraph(108000, images="/home/roeih/SceneGrapher/Data/VisualGenome/data/",
                       imageDataDir=os.path.join("/home/roeih/SceneGrapher/Data/VisualGenome/data", 'by-id/'),
                       synsetFile=os.path.join("/home/roeih/SceneGrapher/Data/VisualGenome/data", 'synsets.json'))

    # Get the data
    data = list(labels)


def create_pre_proccessed_entities_by_xu():
    """
    This function creates entities from the pre-processed data made by Xu et al.
    :return: 
    """

    # Load the pre-processed mapping
    mapping = FilesManager().load_file("{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "VG_SGG_dicts"))
    # Load the VG-SGG.h5
    im_h5 = FilesManager().load_file("{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "meta_preproccesed_data"))
    # Load  the imdb_1024.h5
    imdb = FilesManager().load_file("{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "preproccesed_data"))

    pre_ids_to_object_names = mapping["idx_to_label"]
    pre_ids_to_predicates_names = mapping["idx_to_predicate"]

    Logger().log("Start loading the while data. It will take a while.")

    boxes = list(im_h5["boxes_1024"])
    relationships = list(im_h5["relationships"])
    active_objects = list(im_h5["active_object_mask"])
    object_labels = list(im_h5["labels"])
    predicates_labels = list(im_h5["predicates"])
    image_to_first_rel = list(im_h5["img_to_first_rel"])
    image_to_last_rel = list(im_h5["img_to_last_rel"])
    image_to_first_object = list(im_h5["img_to_first_box"])
    image_to_last_object = list(im_h5["img_to_last_box"])
    split = list(im_h5["split"])
    image_ids = list(imdb["image_ids"])
    image_widths = list(imdb["image_widths"])
    image_heights = list(imdb["image_heights"])
    image_original_widths = list(imdb["original_widths"])
    image_original_heights = list(imdb["original_heights"])

    Logger().log("Start creating the pre-processing entities")
    entities_lst = []

    images = {img.id: img for img in
              GetAllImageData("/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Data/VisualGenome/data/")}

    for image_index in range(len(image_ids)):
        print("Image Index {0}".format(image_index))
        all_objects = {}
        if split[image_index] == 2:

            # image_info
            img_index = image_index
            img_id = image_ids[image_index]
            img_url = images[img_id].url
            img_width = image_widths[image_index]
            img_height = image_heights[image_index]
            img_orig_width = image_original_widths[image_index]
            img_orig_height = image_original_heights[image_index]
            image = ImageMapping(img_id, img_url, img_width, img_height, -1, -1, img_orig_width, img_orig_height,
                                 img_index)

            # objects
            first_object_index = image_to_first_object[image_index]
            last_object_index = image_to_last_object[image_index]

            objects = []
            if first_object_index != -1 or last_object_index != -1:
                for object_index in range(first_object_index, last_object_index + 1):
                    if active_objects[object_index] == False:
                        raise Exception("active_object_mask")

                    obj_x = boxes[object_index][0] - boxes[object_index][2] / 2
                    obj_y = boxes[object_index][1] - boxes[object_index][3] / 2
                    obj_width = boxes[object_index][2]
                    obj_height = boxes[object_index][3]

                    obj_id = object_index
                    obj_names = [pre_ids_to_object_names[str(object_labels[object_index][0])]]
                    obj = ObjectMapping(obj_id, obj_x, obj_y, obj_width, obj_height, obj_names, None, img_url)
                    objects.append(obj)
                    all_objects[object_index] = obj

            # relationships
            first_rel_index = image_to_first_rel[image_index]
            last_rel_index = image_to_last_rel[image_index]
            relationship_lst = []
            if first_rel_index != -1 or last_rel_index != -1:
                for rel_index in range(first_rel_index, last_rel_index + 1):
                    rl_id = rel_index
                    rl_filtered_id = rel_index
                    rl_predicate = pre_ids_to_predicates_names[str(predicates_labels[rel_index][0])]
                    rl_subject = all_objects[relationships[rel_index][0]]
                    rl_object = all_objects[relationships[rel_index][1]]
                    relationship = RelationshipMapping(rl_id, rl_subject, rl_predicate, rl_object, None, img_url,
                                                       rl_filtered_id)
                    relationship_lst.append(relationship)

                    # Create Negatives

            entity = Graph(image, objects, relationship_lst, None)
            entities_lst.append(entity)

            path = os.path.join(PROJECT_ROOT, PREPROCESSED_FILTERED_DATA_SPLIT_PATH, "{0}.p".format(img_id))
            f = open(path, "wb")
            cPickle.dump(entity, f)
            f.close()

    FilesManager().save_file(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "full_filtered_preprocessed_data_test"), entities_lst)


def save_new_images():
    """
    This function load images and save them
    :return: 
    """
    imdb = FilesManager().load_file("{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "preproccesed_data"))
    length = len(imdb['images'])
    images = {img.id: img for img in
              GetAllImageData("/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Data/VisualGenome/data/")}
    for img_ind in range(length):
        print("Image index: {0}".format(img_ind))
        img_id = imdb["image_ids"][img_ind]
        path_lst = images[img_id].url.split('/')
        img = imdb['images'][img_ind]
        img = np.transpose(img, (1, 2, 0))
        img_path = os.path.join(PROJECT_ROOT, VG_DATA_PATH, path_lst[-2], path_lst[-1])
        cv2.imwrite(img_path, img)


def copy_files_from_server():
    """
    This function copies the VG images to another path fo be used
    :return:
    """
    Logger().log("Welcome to copy files")
    # The path
    from_path = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Data/VisualGenome/preprocessed_data"
    to_path = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Temp"

    # The dataset from which the urls will be taken
    train_imgs = cPickle.load(open(
        "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/FilesManager/FeaturesExtraction/PredicatesMaskCNN/Sun_Oct_22_14:15:25_2017/train_set.p"))
    train_imgs = train_imgs[:500]
    test_imgs = cPickle.load(open(
        "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/FilesManager/FeaturesExtraction/PredicatesMaskCNN/Sun_Oct_22_14:15:25_2017/test_set.p"))
    test_imgs = test_imgs[:len(train_imgs) / 3]
    urls_lst = list(set(test_imgs[Detections.Url]))
    Logger().log("Start copying")
    for url in urls_lst:
        suffix = url.split('/')
        create_folder(os.path.join(to_path, suffix[-2]))
        from_path_curr = os.path.join(from_path, suffix[-2], suffix[-1])
        to_path_curr = os.path.join(to_path, suffix[-2], suffix[-1])
        copyfile(from_path_curr, to_path_curr)
        Logger().log("The file from {0} copy to {1}".format(from_path_curr, to_path_curr))


def debug_per_predicate(predicate='', path=""):
    """

    :param predicate:
    :param path:
    :return:
    """
    detections_test = process_to_detections(None, detections_file_name="full_detections_train")
    # detections_test = cPickle.load(open("/home/roeih/SceneGrapher/test_predicts_playing_081117.p"))
    # detection_parser(dir_path=path, files=["/home/roeih/SceneGrapher/test_predicts_playing_081117.p"])
    detections = detections_test[np.where(detections_test[Detections.Predicate] == predicate)]
    for detection in detections:
        url = detection[Detections.Url]
        img_id = int(url.split('/')[-1].split('.')[0])
        img = get_img(url)
        img2 = get_img(url)
        if img is None:
            Logger().log("Image {0} was not found".format(url))
            continue
        # Get object box
        object_box = detection[Detections.ObjectBox]
        subject_box = detection[Detections.SubjectBox]
        subject_label = detection[Detections.SubjectClassifications]
        object_label = detection[Detections.ObjectClassifications]

        # Predicate
        predicate_box = find_union_box(subject_box, object_box)
        predicate_gt_label = detection[Detections.Predicate]
        predicate_predict_label = detection[Detections.UnionFeature]

        # Draw Predicate box with their labels
        VisualizerDrawer.draw_labeled_box(img, predicate_box,
                                          label="<{0}, {1}/{2}, {3}>".format(subject_label,
                                                                             predicate_predict_label,
                                                                             predicate_gt_label,
                                                                             object_label),
                                          rect_color=CvColor.BLUE, scale=500)
        # Draw Object box with their labels
        VisualizerDrawer.draw_labeled_box(img2, object_box,
                                          label=object_label,
                                          rect_color=CvColor.RED, scale=500)
        # Draw Subject box with their labels
        VisualizerDrawer.draw_labeled_box(img2, subject_box,
                                          label=subject_label,
                                          rect_color=CvColor.RED, scale=500)
        file_path = os.path.join(path, "{}_predicates.jpg".format(img_id))
        file_path2 = os.path.join(path, "{}_objects.jpg".format(img_id))
        cv2.imwrite(file_path, img)
        cv2.imwrite(file_path2, img2)


def get_corpus_from_image_captioning():
    """
    This function will get the whole image captioning from the images and saves them as a corpus
    :return:
    """

    # load entities
    # region_interest = GetAllRegionDescriptions("/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/Data/VisualGenome/data")
    region_interest = cPickle.load(open("region_interst10.p"))
    # idx_to_ids = FilesManager().load_file("data.visual_genome.idx_to_ids")
    img_id_to_split = FilesManager().load_file("data.visual_genome.img_id_to_split")

    # index for saving
    ind = 1
    train_corpus = open("train_corpus.txt", "w")
    test_corpus = open("test_corpus.txt", "w")
    Logger().log("Start creating corpus for train and test from image captioning")
    for region in region_interest:
        for phrase_data in region:
            try:

                # Get the phrase
                phrase_txt = phrase_data.phrase.replace(".", "").replace("\n", "").replace(",",
                                                                                           "").rstrip().lstrip() + " "
                img_id = phrase_data.image.id
                Logger().log("Image id: {0}".format(img_id))
                # Write train corpus
                if img_id_to_split[img_id] == 0:
                    train_corpus.write(phrase_txt)

                # Write test corpus
                if img_id_to_split[img_id] == 2:
                    test_corpus.write(phrase_txt)

                ind += 1
            except Exception as e:
                print("Problem with {0} in index: {1} image id: {2}".format(e, ind, img_id))

    Logger().log("Finish to create train and test corpus")
    # Close txt files
    train_corpus.close()
    test_corpus.close()
    return 1


def export_keras_model_tf():
    """
    This function exports Keras Model to tensorflow Model
    :return:
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    img_input = Input(shape=(224, 224, 5), name="image_input")
    # Define ResNet50 model Without Top
    net = ModelZoo()
    model_resnet50 = net.resnet50_with_masking_dual(img_input, trainable=True)
    # model_resnet50 = net.resnet50_base_reg_and_init(img_input, trainable=config.resnet_body_trainable,
    #                                                 kernel_regularizer=regularizers.l2(0.005),
    #                                                 kernel_initializer="he_normal")
    model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
    output_resnet50 = Dense(51, kernel_initializer="he_normal", activation='softmax', name='fc')(
        model_resnet50)

    # Define the model
    model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')


# def shrink_boxes_slow(self, data, heat_map):
#
#     original_detection_centers = {}
#
#     for ind, row in data.iterrows():
#         confidence = row['confidence']
#         box = numpy.asarray([row['x1'], row['y1'], row['x2'], row['y2']])
#         width = box[BOX.X2] - box[BOX.X1]
#         height = box[BOX.Y2] - box[BOX.Y1]
#
#         if self.visualizer:
#             self.visualizer.draw_labeled_box('orig', box, rect_color=CvColor.GREEN,
#                                              label=ind)
#
#         original_detection_centers[ind] = (box[BOX.X1] + width / 2, box[BOX.Y1] + height / 2)
#
#         w_shift = (width * (1 - BOX_RESIZE_FACTOR)) / 2.
#         h_shift = (height * (1 - BOX_RESIZE_FACTOR)) / 2.
#
#         box[BOX.X1] += w_shift
#         box[BOX.X2] -= w_shift
#         box[BOX.Y1] += h_shift
#         box[BOX.Y2] -= h_shift
#         if self.visualizer:
#             self.visualizer.draw_labeled_box('img', box, rect_color=CvColor.GREEN,
#                                              label=ind)
#         width = box[BOX.X2] - box[BOX.X1]
#         height = box[BOX.Y2] - box[BOX.Y1]
#
#         gaussian = numpy.zeros(shape=[height, width], dtype=numpy.float64)
#         gaussian[height / 2, width / 2] = 1
#         gaussian = scipy.ndimage.filters.gaussian_filter(gaussian, (height / 2., width / 2.))
#         cv2.normalize(gaussian, gaussian, 0, confidence, cv2.NORM_MINMAX)
#         heat_map[box[BOX.Y1]:box[BOX.Y2], box[BOX.X1]:box[BOX.X2]] += numpy.expand_dims(gaussian, axis=2)
#
#     return original_detection_centers


def vis_entities():
    """

    :return:
    """

    # Get file list
    files_lst = ["Sat_Nov_11_21:36:12_2017"]
    # files_lst = ["Sat_Nov_11_21:36:12_2017", "Sat_Nov_11_21:38:29_2017", "Sat_Nov_11_21:42:07_2017",
    #              "Sat_Nov_11_21:43:18_2017", "Sat_Nov_11_21:59:10_2017"]
    # Get the entities
    entities_path = FilesManager().get_file_path("data.visual_genome.detections_v4")

    for file_dir in files_lst:
        files = os.listdir(os.path.join(entities_path, file_dir))
        for file_name in files:

            # Load only entities
            if ".log" in file_name:
                continue

            file_path = os.path.join(entities_path, file_dir, file_name)
            file_handle = open(file_path, "rb")
            train_entities = cPickle.load(file_handle)
            file_handle.close()

            i = 0
            for entity in train_entities:

                # if entity.image.id not in [2343115, 2343655, 2343239, 2343622, 2342924]:
                if entity.image.id != 2343239:
                    # print("img_id:{0} with file_name:{1}, filedir:{2}".format(entity.image.id, file_name, file_dir))
                    continue

                # if 3 < len(entity.objects) < 7 and 3 < len(entity.relationships) < 7:
                #     continue

                i += 1
                vis = VisualizerEntity(entity, "/home/roeih/Dropbox/SceneGrapher/Paper_photos")
                confidences = np.array([[0.01, 0.0, 0.05, 0.0, 0.0, 0.03], [0.15, 0.0, .39, .0, .0, .03],
                                        [.82, 1.0, 0.03, 0.98, 0.01, 0.61], [0.01, 0.0, 0.47, 0.01, 0.99, 0.29],
                                        [0.0, 0.0, 0.03, 0.01, 0.0, 0.02], [0.01, 0.0, 0.03, 0.0, 0.0, 0.01]])
                # vis.draw_attention(confidences)
                vis.draw_objects()
                vis.draw_scene_graph()

                vis.save_all()

        print("debug")


def load_NYU_dataset(file_path):
    """

    :return:
    """
    path = os.path.join("/home/roeih/NYU_V2", file_path)
    mat = scipy.io.loadmat(path)


if __name__ == '__main__':
    # Create mini data-set
    # create_data_object_and_predicates_by_img_id()

    file_manager = FilesManager()
    logger = Logger()

    # load_NYU_dataset(file_path="splits.mat")
    vis_entities()
    exit()

    # get_corpus_from_image_captioning()

    # exit()

    # debug_per_predicate(predicate="playing", path="/home/roeih/SceneGrapher/Outputs/Playing")
    # exit()

    # copy_files_from_server()
    # exit()

    # save_new_images()
    # exit()

    # ent = cPickle.load(open("entity_tmp2.p"))
    # exit()

    # create_pre_proccessed_entities_by_xu()
    # exit()

    # split_extracted_data()

    # exit()
    #
    # split_filter_data()

    # exit()
    #
    # visualize_detections(img_id=2415186)
    # exit()
    #
    # find_image_id_via_object("elephant")
    # exit()

    # detection_parser(dir_path="Temp")
    #
    # print("hi")

    # exit()


    logger_parser(dir_path="Temp")

    print("hi")

    exit()

    get_predicates_dict_from_entities()

    exit()

    test_predicted_features(gpu_num=2, objects_training_dir_name="Mon_Jul_24_19:58:35_2017",
                            predicates_training_dir_name="Wed_Aug__2_21:55:12_2017",
                            # predicted_features_dir_name="Tue_Aug__8_23:28:18_2017",
                            predicted_features_dir_name="Wed_Aug__9_10:04:43_2017",
                            predicted_features_file_name="predicated_entities_0_to_1000.p")
    exit()

    create_predicate_count()

    exit()

    Objects_AR_Histogram()

    exit()

    # Filter the data
    filtered_module_data = get_module_filter_data(objects_count_file_name="full_classes_count.p",
                                                  entities_file_name="full_entities.p",
                                                  # entities_file_name="full_entities.p",
                                                  predicates_count_file_name="full_predicates_count.p", nof_objects=150,
                                                  nof_predicates=50, create_negative=True,
                                                  positive_negative_ratio=POSITIVE_NEGATIVE_RATIO)

    exit()
    # Filter mini urls
    get_mini_url()
    exit()

    # Create mini predicate count
    predict_count_dict = create_predicate_count(entities_file_read="mini_final_entities.p",
                                                entities_file_save="mini_predicates_count.p")

    save_union_detections()

    delete_ind_from_detections()

    save_hierarchy_mapping()

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
