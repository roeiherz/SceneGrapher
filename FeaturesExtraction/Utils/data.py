from __future__ import print_function
from graphviz import Digraph
import os
import cPickle
import json
import pprint
import random
import numpy as np
import operator
from Data.VisualGenome.local import GetSceneGraph
from Data.VisualGenome.models import Relationship, RelationshipMapping
from FeaturesExtraction.Lib.PascalVoc import PascalVoc
from FeaturesExtraction.Utils.Boxes import find_union_box
from FeaturesExtraction.Utils.Utils import VG_PATCH_PATH, DATA_PATH, CLASSES_MAPPING_FILE, CLASSES_COUNT_FILE, \
    TRAIN_IMGS_P, VAL_IMGS_P, VisualGenome_PICKLES_PATH, ENTITIES_FILE, HIERARCHY_MAPPING, PascalVoc_PICKLES_PATH, \
    VALIDATION_DATA_SET, TEST_DATA_SET, TRAIN_DATA_SET, VG_VisualModule_PICKLES_PATH, get_mask_from_object, \
    MINI_VG_DATASET_PATH, MINI_IMDB, get_time_and_date, VG_PICKLES_FOLDER_PATH, VisualGenome_DATASETS_PICKLES_PATH, \
    get_img, POSITIVE_NEGATIVE_RATIO, OBJECTS_ALIAS, PREDICATES_ALIAS, PREDICATES_LIST, OBJECTS_LIST, \
    DATA, VISUAL_GENOME, OUTPUTS_PATH, get_bad_urls
from DesignPatterns.Detections import Detections
from FeaturesExtraction.Utils.Visualizer import VisualizerDrawer, CvColor
import cv2
import h5py
import sys

from FilesManager.FilesManager import FilesManager

__author__ = 'roeih'


def create_data_pascal_voc(load=False):
    """
    This function load pickles
    :param load: load field
    :return: train_imgs, val_imgs, class_mapping_test.p, classes_count
    """

    # When loading Pickle
    if load:
        class_mapping = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, CLASSES_MAPPING_FILE), "rb"))
        classes_count = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, CLASSES_COUNT_FILE), "rb"))
        train_imgs = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, TRAIN_IMGS_P), "rb"))
        val_imgs = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, VAL_IMGS_P), "rb"))
        print('loading pickles')
        return train_imgs, val_imgs, class_mapping, classes_count

    pascal_voc = PascalVoc()
    all_imgs, classes_count, class_mapping = pascal_voc.get_data("/home/roeih/PascalVoc/VOCdevkit",
                                                                 pascal_data=['VOC2007'])

    # Add background class
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    # Create json for the class for future use
    with open('classes.json', 'w') as class_data_json:
        json.dump(class_mapping, class_data_json)
    pprint.pprint(classes_count)

    # Shuffle the Data
    random.shuffle(all_imgs)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    return train_imgs, val_imgs, class_mapping, classes_count


def create_mini_data_visual_genome():
    """
    This function creates mini data-set visual-genome
    :return: classes_count, hierarchy_mapping, entities
    """

    # Load from mini-dataset VisualGenome
    imdb = h5py.File(os.path.join(MINI_VG_DATASET_PATH, MINI_IMDB), "r")
    images_id = imdb['image_ids'][:]

    # Comment out some of data for maybe a future use
    # proposals = h5py.File(os.path.join(MINI_VG_DATADET_PATH, "mini_proposals.h5"), "r")
    # rois = h5py.File(os.path.join(MINI_VG_DATADET_PATH, "mini_VG-SGG.h5"), "r")
    # images_heights = imdb["image_heights"][:]
    # images_widths = imdb["image_widths"][:]
    classes_count, hierarchy_mapping, entities = create_data_visual_genome(image_data=None, ids=images_id, load=False,
                                                                           pickles_file_name="mini")
    return classes_count, hierarchy_mapping, entities


def create_data_visual_genome(image_data=None, ids=None, load=False, pickles_file_name=""):
    """
    This function creates or load pickles.
    hierarchy_mapping: dict with mapping between a class label and his object_id
    classes_count: dict with mapping between a class label and the number of his instances in the visual genome data
    :param pickles_file_name: Pickles file name which they will be saved
    :param load: Load from pickle or not
    :param ids: list of image_ids
    :param image_data: image data
    :return: classes_count, hierarchy_mapping, entities
    """

    # Get images ids from image data or from parameter
    if image_data is not None:
        img_ids = [img.id for img in image_data]
    else:
        if ids is None:
            print("Error with loading image ids")
            exit()
        img_ids = ids

    classes_count = {}
    # Map between label to images
    hierarchy_mapping = {}

    if load:
        # Check if pickles are already created
        classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_COUNT_FILE)
        classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_MAPPING_FILE)
        entities_path = os.path.join(VisualGenome_PICKLES_PATH, ENTITIES_FILE)

        if os.path.isfile(classes_count_path) and os.path.isfile(classes_mapping_path) and os.path.isfile(
                entities_path):
            classes_count = cPickle.load(file(classes_count_path, 'rb'))
            hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
            entities = []
            # entities = cPickle.load(file(entities_path, 'rb'))
            return classes_count, hierarchy_mapping, entities

    # Create classes_count, hierarchy_mapping and entities
    entities = []
    # index for saving
    ind = 1
    print("Start creating pickle for VisualGenome Data with changes")
    for img_id in img_ids:
        try:
            entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
                                   synsetFile=DATA_PATH + "synsets.json")
            entities.append(entity)
            objects = entity.objects
            for object in objects:
                obj_id = object.id
                # if len(object.names) > 1:
                #     print("Two labels per object in img_id:{0}, object_id:{1}, labels:{2}".format(img_id, obj_id,
                #                                                                                   object.names))

                label = object.names[0]

                # Update the classes_count dict
                if label in classes_count:
                    # Check if label is already in dict
                    classes_count[label] += 1
                else:
                    # Init label in dict
                    classes_count[label] = 1

                # Update hierarchy_mapping dict
                if label not in hierarchy_mapping:
                    hierarchy_mapping[label] = obj_id

            # Printing Alerting
            if ind % 10000 == 0:
                save_pickles(classes_count, entities, hierarchy_mapping, iter=str(ind))
                # print("This is iteration number: {}".format(ind))
            # Updating index
            ind += 1
            print("This is iteration number: {}".format(img_id))

        except Exception as e:
            print("Problem with {0} in index: {1}".format(e, img_id))

    save_pickles(classes_count, entities, hierarchy_mapping, iter=pickles_file_name)
    return classes_count, hierarchy_mapping, entities


def save_pickles(classes_count, entities, hierarchy_mapping, iter=''):
    """
    This function save the pickles each iter
    :param classes_count: dict the classes count
    :param entities: dict with the entities
    :param hierarchy_mapping: dict with hierarchy mapping
    :param iter: string iteration number
    """
    # Save classes_count file
    classes_count_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + CLASSES_COUNT_FILE), 'wb')
    # Pickle classes_count
    cPickle.dump(classes_count, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Save hierarchy_mapping file
    hierarchy_mapping_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + CLASSES_MAPPING_FILE), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    # Save entities list
    entities_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + ENTITIES_FILE), 'wb')
    # Pickle entities
    cPickle.dump(entities, entities_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    entities_file.close()


def load_pickles(classes_mapping_path, classes_count_path, entities_path):
    """
    This function save the pickles each iter
    :param classes_count_path: classes_count file name
    :param entities_path: entities file name
    :param classes_mapping_path: hierarchy_mapping file name
    :return classes_count, hierarchy_mapping and entities
    """
    classes_count = cPickle.load(open(classes_count_path, 'rb'))
    hierarchy_mapping = cPickle.load(open(classes_mapping_path, 'rb'))
    entities = cPickle.load(open(entities_path, 'rb'))
    return classes_count, hierarchy_mapping, entities


def get_sorted_data(classes_count_file_name="final_classes_count.p",
                    hierarchy_mapping_file_name="final_class_mapping.p", entities_file_name="final_entities.p",
                    nof_labels=150):
    """
    This function his sorted the hierarchy_mapping and classes_count by the number of labels
    :param entities_file_name: the full entities of *all* the dataset
    :param nof_labels: Number of labels
    :param classes_count_file_name: classes count of *all* the dataset
    :param hierarchy_mapping_file_name: hierarchy_mapping of *all* the dataset
    :return:  a dict of classes_count (mapping between the class and its instances), a dict of hierarchy_mapping
            (mapping between the class and its object id), entities
    """

    # Check if pickles are already created
    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, classes_count_file_name)
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, hierarchy_mapping_file_name)
    entities_path = os.path.join(VisualGenome_PICKLES_PATH, entities_file_name)

    if os.path.isfile(classes_count_path) and os.path.isfile(classes_mapping_path) and os.path.isfile(entities_path):
        print(
            'Files are already exist {0}, {1} and {2}'.format(classes_count_path, classes_mapping_path, entities_path))
        classes_count = cPickle.load(file(classes_count_path, 'rb'))
        hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
        entities = np.array(cPickle.load(file(entities_path, 'rb')))
        return classes_count, hierarchy_mapping, entities

    # Sort and pre-process the 3 pickle files

    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, classes_count_file_name)
    # Load the frequency of labels
    classes_count = cPickle.load(file(classes_count_path, 'rb'))
    # Sort classes_count by value
    sorted_classes_count = sorted(classes_count.items(), key=operator.itemgetter(1), reverse=True)
    # Get the most frequent 150 labels
    top_sorted_class = sorted_classes_count[:nof_labels]
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, hierarchy_mapping_file_name)
    # Load the full hierarchy_mapping
    hierarchy_mapping_full = cPickle.load(file(classes_mapping_path, 'rb'))

    # Create a new hierarchy_mapping for top labels
    hierarchy_mapping = {}
    top_sorted_class_keys = [classes[0] for classes in top_sorted_class]
    for key in hierarchy_mapping_full:
        if key in top_sorted_class_keys:
            hierarchy_mapping[key] = hierarchy_mapping_full[key]

    # Save hierarchy_mapping file for only the top labels
    hierarchy_mapping_file = file(classes_mapping_path, 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    # Save classes_count file for only the top labels
    classes_count_file = file(classes_count_path, 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(dict(top_sorted_class), classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Load entities pickle
    entities = np.array(cPickle.load(file(entities_path, 'rb')))
    return classes_count, hierarchy_mapping, entities


def splitting_to_datasets(entities, training_percent, testing_percent, num_epochs,
                          path=VisualGenome_DATASETS_PICKLES_PATH, config=None):
    """
    This function splits the data for train and test dataset
    :param config: config
    :param path: path where we are saving the data
    :param num_epochs: number of epochs
    :param testing_percent: testing percent from the data
    :param training_percent: training percent from the data
    :param entities: entities from visual genome
    :return: list of entities of train and test data
    """

    # Load datasets from cache
    if config is not None and config.use_cache_dir:
        train_dataset_path = os.path.join(config.loading_model_folder, TRAIN_DATA_SET)
        test_dataset_path = os.path.join(config.loading_model_folder, TEST_DATA_SET)
        validation_dataset_path = os.path.join(config.loading_model_folder, VALIDATION_DATA_SET)
        print("Loading cached data-sets: training-{0}, testing-{1} and valiation-{2}".format(train_dataset_path,
                                                                                             test_dataset_path,
                                                                                             validation_dataset_path))
        train_imgs = cPickle.load(open(train_dataset_path, 'rb'))
        test_imgs = cPickle.load(open(test_dataset_path, 'rb'))
        val_imgs = cPickle.load(open(validation_dataset_path, 'rb'))

        print("Debug printing- the number of train samples: {0}, the number of test samples: {1}, "
              "the number of validation samples: {2}".format(len(train_imgs), len(test_imgs), len(val_imgs)))

        return train_imgs, test_imgs, val_imgs

    number_of_samples = len(entities)
    train_size = int(number_of_samples * training_percent)
    test_size = int(number_of_samples * testing_percent)
    validation_size = number_of_samples - (train_size + test_size)

    if not train_size + test_size + validation_size == number_of_samples:
        error_msg = 'Data size of (train + test + validation) is {0} and should be number of labels: {1}'.format(
            train_size + test_size + validation_size, number_of_samples)
        print(error_msg)
        raise Exception(error_msg)

    # Create a numpy array of indices of the data
    indices = np.arange(len(entities))
    # Shuffle the indices of the data
    random.shuffle(indices)

    # Get the train + test + val dataset
    train_imgs = entities[indices[:train_size]]
    test_imgs = entities[indices[train_size:train_size + test_size]]
    val_imgs = entities[indices[train_size + test_size:]]

    # Take the round number of each dataset per the number of epochs
    # num_of_samples_per_train_updated = len(train_imgs) / num_epochs * num_epochs
    # train_imgs = train_imgs[:num_of_samples_per_train_updated]
    # num_of_samples_per_test_updated = len(test_imgs) / num_epochs * num_epochs
    # test_imgs = test_imgs[:num_of_samples_per_test_updated]
    # num_of_samples_per_val_updated = number_of_samples - num_of_samples_per_train_updated - num_of_samples_per_test_updated
    # val_imgs = val_imgs[:num_of_samples_per_val_updated]

    # print("Debug printing- the number of train samples: {0}, the number of test samples: {1}, "
    #       "the number of validation samples: {2}".format(num_of_samples_per_train_updated,
    #                                                      num_of_samples_per_test_updated,
    #                                                      num_of_samples_per_val_updated))

    # Save train-set and test-set and validation-set
    pickle_dataset(train_imgs, test_imgs, val_imgs, path)
    return train_imgs, test_imgs, val_imgs


def pickle_dataset(train_set, test_set, validation_set, path):
    """
    This function save the data-set, test-set and validation-set to pickles
    :param path: path where we are saving the data
    :param train_set: the train data-set
    :param test_set: the test data-set
    :param validation_set: the validation data-set
    """

    train_set_filename = open(os.path.join(path, TRAIN_DATA_SET), 'wb')
    # Pickle classes_count
    cPickle.dump(train_set, train_set_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    train_set_filename.close()
    # Save hierarchy_mapping file
    test_set_filename = open(os.path.join(path, TEST_DATA_SET), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(test_set, test_set_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    test_set_filename.close()
    # Save entities list
    validation_set_filename = open(os.path.join(path, VALIDATION_DATA_SET), 'wb')
    # Pickle entities
    cPickle.dump(validation_set, validation_set_filename, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    validation_set_filename.close()

    print("Debug printing- the number of train samples: {0}, the number of test samples: {1}, "
          "the number of validation samples: {2}".format(len(train_set), len(test_set), len(validation_set)))


def generate_new_hierarchy_mapping(hierarchy_mapping):
    """
    This function generates a new hierarchy mapping from index 0 to number of classes
    :param hierarchy_mapping: a dict with mapping between label string to an object id from visual genome dataset
    :return: new dict with mapping between label string and a new count from 0 to number of classes
    """

    ind = 0
    new_hierarchy_mapping = {}
    for label in hierarchy_mapping.keys():
        new_hierarchy_mapping[label] = ind
        ind += 1

    return new_hierarchy_mapping


def get_predicate_hierarchy_mapping_from_detections(detections, path, config=None):
    """
    This function get the predicate hierarchy mapping from detections
    :param detections: a Detections numpy dtype
    :param config: config
    :param path: saving or loading the classes_count_per_objects and hierarchy_mapping_per_objects from path folder
    :return: a new dict of hierarchy mapping of predicate (new_hierarchy_mapping)
            and a new dict of number of classes (new_classes_count)
    """

    # Load hierarchy mapping and class counting from cache
    if config is not None and config.use_cache_dir:
        classes_count_path = os.path.join(config.loading_model_folder, CLASSES_COUNT_FILE)
        hierarchy_mapping_path = os.path.join(config.loading_model_folder, CLASSES_MAPPING_FILE)
        print("Loading from cached hierarchy mapping from {0} and class counting {1}".format(hierarchy_mapping_path,
                                                                                             classes_count_path))
        classes_count_per_objects = cPickle.load(open(classes_count_path, 'rb'))
        hierarchy_mapping_per_objects = cPickle.load(open(hierarchy_mapping_path, 'rb'))
        return classes_count_per_objects, hierarchy_mapping_per_objects

    ind = 0
    new_hierarchy_mapping = {}
    new_classes_count = {}
    predicates = detections[Detections.Predicate]

    # # Make the predicates to set
    # predicates_set = set(list(predicates))
    for predicate in predicates:

        # Update the new_hierarchy_mapping
        if predicate not in new_hierarchy_mapping:
            new_hierarchy_mapping[predicate] = ind
            ind += 1

        # Update the new_classes_count
        if predicate not in new_classes_count:
            new_classes_count[predicate] = 1
        else:
            new_classes_count[predicate] += 1

    # Save classes_count file
    classes_count_file = file(os.path.join(path, CLASSES_COUNT_FILE), 'wb')
    # Pickle classes_count
    cPickle.dump(new_classes_count, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Save hierarchy_mapping file
    hierarchy_mapping_file = file(os.path.join(path, CLASSES_MAPPING_FILE), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(new_hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    return new_classes_count, new_hierarchy_mapping


def process_to_detections(relations, detections_file_name="detections.p", debug=False):
    """
    This method embeddings relations to detections
    :param debug: debug flag for printing subject box, object box and the union of them
    :param detections_file_name: detections pickle file name
    :param relations: a numpy array of relationships
    :return: numpy detections array
    """
    filemanager = FilesManager()

    detections_path_token = "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(detections_file_name))

    # Check if pickles are already created
    detections_path = filemanager.get_file_path(detections_path_token)

    if os.path.isfile(detections_path):
        print('File is already exist {0}'.format(detections_path))
        detections = filemanager.load_file(detections_path_token)
        return detections

    bad_urls = get_bad_urls()
    detections = Detections(len(relations))
    id = 0
    for relation in relations:

        # Sorting bad urls
        if relation.url in bad_urls:
            continue

        # Update Relation Id
        detections[id][Detections.Id] = relation.filtered_id
        # Update Subject Id
        detections[id][Detections.SubjectId] = relation.subject.id
        # Get the mask: a dict with {x1,x2,y1,y2}
        mask_subject = get_mask_from_object(relation.subject)
        # Saves as a box
        subject_box = np.array([mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])
        # Update Subject Box
        detections[id][Detections.SubjectBox] = subject_box
        # Update Object Id
        detections[id][Detections.ObjectId] = relation.object.id
        # Get the mask: a dict with {x1,x2,y1,y2}
        mask_object = get_mask_from_object(relation.object)
        # Saves as a box
        object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
        # Update Object box
        detections[id][Detections.ObjectBox] = object_box
        # Update Subject Classification
        detections[id][Detections.SubjectClassifications] = relation.subject.names[0]
        # Update Object Classification
        detections[id][Detections.ObjectClassifications] = relation.object.names[0]
        # Update Url
        detections[id][Detections.Url] = relation.url
        # Update Predicate
        detections[id][Detections.Predicate] = relation.predicate

        detections[id][Detections.UnionBox] = find_union_box(subject_box, object_box)

        # region debug of printing the subject box, object box and the union of them
        if debug:
            img = get_img(relation.url)
            VisualizerDrawer.draw_labeled_box(img, np.array(
                [mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']]),
                                              label=relation.subject.names[0],
                                              rect_color=CvColor.GREEN,
                                              scale=2000)

            VisualizerDrawer.draw_labeled_box(img, np.array(
                [mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']]),
                                              label=relation.object.names[0],
                                              rect_color=CvColor.BLUE,
                                              scale=2000)

            VisualizerDrawer.draw_labeled_box(img, find_union_box(subject_box, object_box),
                                              label='union',
                                              rect_color=CvColor.BLACK,
                                              scale=2000)

            cv2.imwrite("test{}.jpg".format(id), img)
        # endregion

        # Update index
        id += 1

    # Pickle detections numpy Detection dtype
    filemanager.save_file(detections_path_token, detections)

    return detections


def preprocess_entities_by_mapping(entities, objects_alias_mapping, predicates_alias_mapping):
    """
    This function pre-process the entities by object alias mapping and predicate
    :param entities: list of entities
    :param objects_alias_mapping: objects dictionary mapping between "old object: new object"
    :param predicates_alias_mapping: predicates dictionary mapping between "old predicate: new predicate"
    :return: list of entities after their objects and predicates have been changed according to new alias mapping
    """

    for entity in entities:

        for object in entity.objects:
            candidate_object = object.names[0].lower()

            # Update object name according to the objects_alias_mapping or just save it lower-case
            if candidate_object in objects_alias_mapping:
                object.names[0] = objects_alias_mapping[candidate_object]
            else:
                object.names[0] = candidate_object

        for relation in entity.relationships:
            candidate_predicate = relation.predicate.lower()

            # Update object name according to the predicates_to_be_used or just save it lower-case
            if candidate_predicate in predicates_alias_mapping:
                relation.predicate = predicates_alias_mapping[candidate_predicate]
            else:
                relation.predicate = candidate_predicate


def get_module_filter_data(objects_count_file_name="mini_classes_count.p", entities_file_name="full_entities.p",
                           predicates_count_file_name="mini_predicates_count.p", nof_objects=150, nof_predicates=50,
                           create_negative=False, positive_negative_ratio=POSITIVE_NEGATIVE_RATIO):
    """
    This function filtered the entities data by top num of objects and top number of predicates 
    :return: filtered_module_data which is a dict with entities, hierarchy mapping of objects and hierarchy mapping of 
                predicates
    """

    filemanager = FilesManager()

    # Load Objects alias
    objects_alias_filename = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(OBJECTS_ALIAS)))
    objects_alias_mapping, objects_alias_words_target = make_alias_dict(objects_alias_filename)

    # Load Predicates alias
    predicates_alias_filename = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(PREDICATES_ALIAS)))
    predicates_alias_mapping, predicates_alias_words_target = make_alias_dict(predicates_alias_filename)

    # Load Objects list
    objects_list_filename = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(OBJECTS_LIST)))
    objects_to_be_used = make_list(objects_list_filename)

    # Load Predicates list
    predicates_list_filename = filemanager.get_file_path(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(PREDICATES_LIST)))
    predicates_to_be_used = make_list(predicates_list_filename)

    # Load entities
    entities = np.array(filemanager.load_file(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(entities_file_name))))

    # PreProcess Objects by Mapping
    preprocess_entities_by_mapping(entities, objects_alias_mapping, predicates_alias_mapping)

    # regionOld code in which we calculate from the top counts data the predicates_to_be_used and the objects_to_be_used
    # # Load object counts dict
    # objects_count_path = os.path.join(VisualGenome_PICKLES_PATH, objects_count_file_name)
    # objects_count = cPickle.load(file(objects_count_path, 'rb'))
    #
    # # Load predicates counts dict
    # predicates_count_path = os.path.join(VisualGenome_PICKLES_PATH, predicates_count_file_name)
    # predicate_count = cPickle.load(file(predicates_count_path, 'rb'))
    # Sorting - Extracting the most popular predicates from a predicate counter dictionary
    # sorted_predicates_count = sorted(predicate_count.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_predicates = sorted_predicates_count[:nof_predicates]
    # predicates_to_be_used = dict(sorted_predicates)
    #
    # # Sorting - Extracting the most popular objects from a objects counter dictionary
    # sorted_objects_count = sorted(objects_count.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_objects = sorted_objects_count[:nof_objects]
    # objects_to_be_used = dict(sorted_objects)
    # endregion

    # Counts index for relationship id
    relation_ind = 0
    object_ind = 0
    total_object = 0
    total_relations = 0
    total_negatives = 0
    relation_id = 10000000
    entity_curr = 0
    for entity in entities:
        objects_filtered = []

        for object in entity.objects:
            total_object += 1
            # Filter out object
            if not object.names[0] in objects_to_be_used:
                continue

            objects_filtered.append(object)
            object_ind += 1

        # Rewrite objects
        entity.objects = objects_filtered[:]

        relationship_filtered = []
        for relation in entity.relationships:

            # Filter out object
            if not relation.subject.names[0] in objects_to_be_used:
                continue
            if not relation.object.names[0] in objects_to_be_used:
                continue

            # Filter out predicate
            if not relation.predicate in predicates_to_be_used:
                continue

            # New filtered Id
            relation.filtered_id = relation_ind
            # Increment id
            relation_ind += 1
            relationship_filtered.append(relation)

        # Rewrite relations - the slice is for copy the list
        entity.relationships = relationship_filtered[:]
        # Increment
        entity_curr += 1

        # Check the relationship_filtered list is not empty
        if create_negative and relationship_filtered:
            # Create Negatives
            negative_relations, relation_ind, relation_id = create_negative_relations(entity, relation_id, relation_ind,
                                                                                      positive_negative_ratio=positive_negative_ratio)
            # Rewrite relationships
            entity.relationships += negative_relations
            total_negatives += len(negative_relations)
            # Print
            print("Number of (negatives, positive) relations ({0}, {1}) in Entity number: {2}".format(
                len(negative_relations), len(relationship_filtered), entity_curr))
        else:
            print("Warning: No relations in Entity: {}".format(entity_curr))

        total_relations += len(entity.relationships)

    print("Number of filtered relations: {}".format(relation_ind))
    print("Number of filtered objects: {}".format(object_ind))
    print("Number of total objects: {}".format(total_object))
    print("Number of total relations: {}".format(total_relations))
    print("Number of total negatives: {}".format(total_negatives))

    object_ids = {}
    id = 0
    for object in objects_to_be_used:
        object_ids[object] = id
        id += 1

    predicate_ids = {}
    id = 0
    for predicate in predicates_to_be_used:
        predicate_ids[predicate] = id
        id += 1

    # Add negative id
    predicate_ids[u'neg'] = id

    # Create new filtered data
    filtered_module_data = {"object_ids": object_ids, "predicate_ids": predicate_ids, "entities": entities,
                            'entities_module': entities[:len(entities) / 2],
                            "entities_visual_module": entities[len(entities) / 2:]}

    # Save filtered_module_data file for only the top labels
    # filemanager.save_file(
    #     "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "mini_filtered_data"), filtered_module_data)
    filemanager.save_file(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, "full_filtered_data"), filtered_module_data)

    return filtered_module_data


def preprocess_data_by_mapping(dicts_count, alias_mapping):
    """
    This function preprocessed the objects data according to the objects_alias_mapping
    :param dicts_count: a dict with object and number of its instances
    :param alias_mapping: a dict between object and its correct object
    :return: a new dictionary after aliasing the keys
    """

    new_mapping_dict = {}
    for (key, value) in dicts_count.items():

        # Replace the key with lower string
        new_key = unicode.lower(key)

        # Replace the key with object alias
        if new_key in alias_mapping:
            new_key = alias_mapping[new_key]

        # Check if the object key is already exists in the dict
        if new_key in new_mapping_dict:
            new_mapping_dict[new_key] += value
        else:
            new_mapping_dict[new_key] = value

    return new_mapping_dict


def create_negative_relations(entity, relation_id, filtered_id, positive_negative_ratio=POSITIVE_NEGATIVE_RATIO):
    """
    Find negative relations by checking that subject and object that don't have a familiar relation
    :param filtered_id: the filtered id which is a unique and incremental id that we give for each relation
    :param positive_negative_ratio: positive to negative ratio. default is 3
    :param entity: entity object of image in Visual Genome data-set
    :param relation_id: a new relation_id for negative relationship
    :return: negative_relations is a list of a negative relations , filtered_id is the id which we incremented
    for-each relation and relation_id is the fake id which we give for each relation
    """
    # Shuffle entities
    objects = np.copy(entity.objects)
    np.random.shuffle(objects)

    # Create random (subject, object) tuples
    sub_obj_lst = create_random_tuples(entity, positive_negative_ratio)

    negative_relations = []
    # We are create negatives with a positive negative ratio
    for tup in sub_obj_lst:

        # Get the (subject, object) tuple
        subject = objects[tup[0]]
        object = objects[tup[1]]

        # Make sure it isn't the same object
        if object == subject:
            print('Strange')
            continue

        negative_flag = True
        # Check if object and subject are not already in
        for relation in entity.relationships:

            # Negative is a subject and object that don't have a relation
            # Check if the subject or object are in relations.
            if relation.subject == subject and relation.object == object:
                negative_flag = False
                break

            if relation.subject == object and relation.object == subject:
                negative_flag = False
                break

        # If subject and object that don't have a relation then it can be updated to negative relation
        if negative_flag:

            update = True
            # Check we don't have a symmetry negative relations
            for relation in negative_relations:
                if relation.subject == subject and relation.object == object:
                    update = False
                    break

                if relation.subject == object and relation.object == subject:
                    update = False
                    break

            # Update for a new negative relation
            if update:
                neg_relation = RelationshipMapping(id=relation_id, subject=subject, predicate="neg", object=object,
                                                   synset=[], url=entity.image.url, filtered_id=filtered_id)
                # New filtered Id
                neg_relation.filtered_id = filtered_id
                negative_relations.append(neg_relation)
                relation_id += 1
                filtered_id += 1

            # Check if are in the positive negative ratio
            if len(negative_relations) >= len(entity.relationships) * positive_negative_ratio:
                return negative_relations, filtered_id, relation_id

    return negative_relations, filtered_id, relation_id


def create_random_tuples(entity, positive_negative_ratio):
    """
    This function create random tuples with the same tuple and without symmetry
    :param entity: entity Visual Genome
    :param positive_negative_ratio: the threshold which we check if we created enough tuples
    :return: list of tuples
    """
    # List of object and subject tuples of pairs
    sub_obj_lst = []

    for i in range(len(entity.objects)):
        for j in range(len(entity.objects)):
            # Don't generate the same number in the tuple e.g (5,5)
            if i == j:
                continue

            # Don't generate the symmetry tuple in the tuple e.g (7,5)
            if (j, i) in sub_obj_lst:
                continue

            # Create new tuple
            new_tup = (i, j)

            # Append (subject, object) tuple only if it not exits
            if new_tup not in sub_obj_lst:
                sub_obj_lst.append(new_tup)

    # Check the length of number of tuples
    if len(sub_obj_lst) < len(entity.relationships) * positive_negative_ratio:
        print(
            "Warning: too few (subject, object) tuples than expected - {0} - in entity {1}. \nJust {2} number of objects and {3} number of relations".format(
                len(sub_obj_lst), entity.image.url, len(entity.objects), len(entity.relationships)))

    # Shuffle the data
    np.random.shuffle(sub_obj_lst)
    return sub_obj_lst


def get_filtered_data(filtered_data_file_name="filtered_module_data.p", category='entities_visual_module',
                      load_entities=True):
    """
    This function loads a dict that was created by get_module_filter_data function.
    The dict contains:
    * filtered entities by the top 150 objects and top 50 predicates 
    * hierarchy mapping of objects  
    * hierarchy mapping of predicates
    :param load_entities: A flag to load entities or return None
    :param category: category is 'entities_visual_module' (only second - 1/2 entities) or 'entities' (all entities) or 
            'entities_module' (only first - 1/2 entities/ shiko)
    :param filtered_data_file_name: the file name of the filtered data
    :return: entities, hierarchy mapping of objects and hierarchy mapping of predicates
    """

    filemanager = FilesManager()

    # Load the filtered file
    filtered_module_data = filemanager.load_file(
        "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(filtered_data_file_name)))

    if load_entities:
        entities = list(filtered_module_data[category])
    else:
        entities = None

    hierarchy_mapping_objects = dict(filtered_module_data['object_ids'])
    hierarchy_mapping_predicates = dict(filtered_module_data['predicate_ids'])

    # Delete the whole data, is no longer needed
    del filtered_module_data
    return entities, hierarchy_mapping_objects, hierarchy_mapping_predicates


def make_alias_dict(file_name):
    """
    This function create an alias dictionary from a txt file
    :param file_name: file name
    :return: dictionary and list
    """

    # Dict mapping between words->target words
    dict_mapping = {}
    # List of the target words
    words_target = []
    for line in open(file_name, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in dict_mapping else dict_mapping[alias[0]]
        for a in alias:
            dict_mapping[a] = alias_target  # use the first term as the aliasing target
        words_target.append(alias_target)
    return dict_mapping, words_target


def make_list(list_file):
    """
    Create a blacklist list from a file
    :param list_file:
    :return:
    """
    return [line.strip('\n').strip('\r') for line in open(list_file)]


def get_name_from_file(filename):
    """
    This function get the name from the filename (from example.txt we returned example)
    :param filename: the file name including the extension .txt
    :return: only the name of the filename
    """
    return filename.split(".")[0]


def visualize_detections(detections=None, relations=None, path_to_save=OUTPUTS_PATH, img_id=""):
    """
    This function
    :param img_id: image id
    :param path_to_save: path to save the output folder
    :param relations:
    :param detections: The detections to be visualized
    :return:
    """

    def create_folder(path):
        """
        Checks if the path exists, if not creates it.
        :param path: A valid path that might not exist
        :return: An indication if the folder was created
        """
        folder_missing = not os.path.exists(path)

        if folder_missing:
            # Using makedirs since the path hierarchy might not fully exist.
            try:
                os.makedirs(path)
            except OSError as e:
                if (e.errno, e.strerror) == (17, 'File exists'):
                    print(e)
                else:
                    raise

            print('Created folder {0}'.format(path))

        return folder_missing

    detections = cPickle.load(open("{}_detections.p".format(img_id)))
    # Get image
    img_objects = get_img(detections[0][Detections.Url])
    img_relations = get_img(detections[0][Detections.Url])
    img_relations_pos = get_img(detections[0][Detections.Url])
    img_relations_triplets = get_img(detections[0][Detections.Url])

    for detection in detections:

        # Get the subject box
        draw_subject_box = detection[Detections.SubjectBox]
        # Get Predicted Subject
        subject_predict = detection[Detections.PredictSubjectClassifications]
        # Get Subject GT
        subject_gt = detection[Detections.SubjectClassifications]
        # Define Subject color
        subject_color = CvColor.BLUE
        if subject_gt != subject_predict:
            subject_color = CvColor.BLACK

        if subject_predict == "person":
            subject_color = CvColor.RED

        # Draw Subject box with their labels
        VisualizerDrawer.draw_labeled_box(img_objects, draw_subject_box, label=subject_predict + "/" + subject_gt,
                                          rect_color=subject_color, scale=500)

        # Get the subject box
        draw_object_box = detection[Detections.ObjectBox]
        # Get Predicted Object
        object_predict = detection[Detections.PredictObjectClassifications]
        # Get Object GT
        object_gt = detection[Detections.ObjectClassifications]
        # Define Subject color
        object_color = CvColor.GREEN
        if object_gt != object_predict:
            object_color = CvColor.BLACK

        if object_predict == "person":
            object_color = CvColor.RED

        # Draw Object box with their labels
        VisualizerDrawer.draw_labeled_box(img_objects, draw_object_box, label=object_predict + "/" + object_gt,
                                          rect_color=object_color, scale=500)

        # Get the predicate box
        draw_predicate_box = detection[Detections.UnionBox]
        # Get Predicted Predicate
        predicate_predict = detection[Detections.UnionFeature]
        # Get Predicate GT
        predicate_gt = detection[Detections.Predicate]
        # Define Subject color
        predicate_color = CvColor.GREEN
        if predicate_predict != predicate_gt:
            predicate_color = CvColor.BLACK

        if predicate_predict == "has":
            predicate_color = CvColor.RED

        if predicate_gt != "neg":
            # Draw Predicate box with their labels
            VisualizerDrawer.draw_labeled_box(img_relations_pos, draw_predicate_box,
                                              label=predicate_predict + "/" + predicate_gt, rect_color=predicate_color,
                                              scale=500)
            # Draw Predicate box with their labels
            VisualizerDrawer.draw_labeled_box(img_relations_triplets, draw_predicate_box,
                                              label="<{0}, {1}, {2}>/".format(subject_predict, predicate_predict,
                                                                              object_predict),
                                              label2="<{0}, {1}, {2}>".format(subject_gt, predicate_gt, object_gt),
                                              rect_color=predicate_color, scale=500)

        # Draw Predicate box with their labels
        VisualizerDrawer.draw_labeled_box(img_relations, draw_predicate_box,
                                          label=predicate_predict + "/" + predicate_gt, rect_color=predicate_color,
                                          scale=500)

    # Get time and date
    time_and_date = get_time_and_date()

    # Path for the training folder
    path = os.path.join(path_to_save, time_and_date)
    # Create a new folder for training
    create_folder(path)
    cv2.imwrite(os.path.join(path, "objects_img_{}.jpg".format(img_id)), img_objects)
    cv2.imwrite(os.path.join(path, "predicates_all_img_{}.jpg".format(img_id)), img_relations)
    cv2.imwrite(os.path.join(path, "predicates_pos_img_{}.jpg".format(img_id)), img_relations_pos)
    cv2.imwrite(os.path.join(path, "predicates_triplets_img_{}.jpg".format(img_id)), img_relations_triplets)


def draw_graph(only_gt, pred_gt, pred, obj_gt, obj, predicate_ids, object_ids, filename):
    """
    create a pdf of the scene graph
    :param only_gt: Means we print ONLY GT without prediction
    :param pred_gt: 2D array - ground truth label for every objects i and j
    :param pred: 2D array - predicted label for every objects i and j
    :param obj_gt: 1D array - ground truth label for every object i
    :param obj: 1D array - predicted label for every object i
    :param predicate_ids: predicate dictionary - label id to label name
    :param object_ids: object dictionary - label id to label name
    :param filename: file name to save
    """
    # create the graph
    u = Digraph('sg', filename='sg.gv', format="svg")
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    u.node_attr.update(style='filled')

    # create objects nodes
    for obj_index in range(obj_gt.shape[0]):
        if only_gt:
            name = object_ids[object_ids.keys().index(obj_gt[obj_index])]
        else:
            name = object_ids[object_ids.keys().index(obj[obj_index])] + "/" + \
                   object_ids[object_ids.keys().index(obj_gt[obj_index])]

        u.node(str(obj_index), label=name, color='lightblue2')

    # create predicates nodes and edges
    for sub_index in range(pred_gt.shape[0]):
        for obj_index in range(pred_gt.shape[1]):
            if pred_gt[sub_index][obj_index] != 50:
                edge_key = '%s_%s' % (sub_index, obj_index)

                if only_gt:
                    name = predicate_ids[predicate_ids.keys().index(pred_gt[sub_index][obj_index])]
                else:
                    name = predicate_ids[predicate_ids.keys().index(pred[sub_index][obj_index])] + \
                           "/" + predicate_ids[predicate_ids.keys().index(pred_gt[sub_index][obj_index])]

                u.node(edge_key, label=name, color='red')
                u.edge(str(sub_index), edge_key)
                u.edge(edge_key, str(obj_index))
    # u.save(filename, ".")
    u.render(filename=filename)


def word2vec_mapping_func():
    """
    This function returns a dict with a mapping between relations
    :return:
    """
    return {"belonging to": "belonging", "parked on": "parked", "growing on": "growing", "standing on": "standing",
            "made of": "made", "attached to": "attached", "hanging from": "hanging", "in front of": "front",
            "lying on": "lying", "flying in": "flying", "looking at": "looking", "on back of": "back",
            "laying on": "laying", "walking on": "walking", "walking in": "walking", "sitting on": "sitting",
            "covered in": "covered", "part of": "part", "painted on": "painted", "mounted on": "mounted"}
