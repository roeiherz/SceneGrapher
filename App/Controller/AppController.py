import traceback
import math

import cPickle
from Data.VisualGenome.models import ObjectMapping
from FeaturesExtraction.Utils.Boxes import find_union_box
from FeaturesExtraction.Utils.Utils import get_img, get_mask_from_object, PROJECT_ROOT, FILTERED_DATA_SPLIT_PATH, \
    EXTRACTED_DATA_SPLIT_PATH
from FeaturesExtraction.Utils.Visualizer import VisualizerDrawer, CvColor
from FeaturesExtraction.Utils.data import get_filtered_data, draw_graph
import numpy
import cv2
import os

__author__ = 'roeih'


class AppController(object):
    """
    This class represents a Controller
    """

    SPLIT_REL_PARAM = 10

    def __init__(self):
        """
        Init the Class of AppController
        """

        # Load hierarchy mappings
        _, self.hierarchy_mapping_objects, self.hierarchy_mapping_predicates = get_filtered_data(
            filtered_data_file_name="mini_filtered_data",
            category='entities_visual_module')

        self.objects_index_labels = {label: index for index, label in self.hierarchy_mapping_objects.iteritems()}
        self.predicates_index_labels = {label: index for index, label in self.hierarchy_mapping_predicates.iteritems()}
        self.object_ids_to_object_index_mapping = {}

    def load_entity(self, img_id):
        """"
        This function load entities
        :type img_id: int of image id
        """
        print('Start Loading Entity.')
        path = os.path.join(PROJECT_ROOT, FILTERED_DATA_SPLIT_PATH, "{0}.p".format(img_id))

        # If img_id pickle is not exist return False
        if not os.path.exists(path):
            return None

        # Open img_id pickle
        with open(path) as fl:
            entity = cPickle.load(fl)

        print('Finished Loading Entity. \n')
        return entity

    def load_preprocessed_entity(self, img_id):
        """"
        This function load pre-processed entity - after Feature Extraction Module
        :type img_id: int of image id
        """
        print('Start Loading Entity.')
        path = os.path.join(PROJECT_ROOT, EXTRACTED_DATA_SPLIT_PATH, "{0}.p".format(img_id))
        # todo: debug
        # path = "/home/roeih/SceneGrapher/App/Controller/test_entity_{0}.p".format(img_id)

        # If img_id pickle is not exist return False
        if not os.path.exists(path):
            return None

        # Open img_id pickle
        with open(path) as fl:
            entity = cPickle.load(fl)

        print('Finished Loading Entity. \n')
        return entity

    def get_objects_hierarchy_mapping(self):
        """
        This function returns Objects Hierarchy Mappings
        :return:
        """
        return self.hierarchy_mapping_objects

    def get_predicates_hierarchy_mapping(self):
        """
        This function returns Predicates Hierarchy Mappings
        :return:
        """
        return self.hierarchy_mapping_predicates

    @staticmethod
    def from_object_to_objects_mapping(objects, correct_labels, url):
        """
        This function get objects from entities and transforms it to object mapping list
        :param objects: the objects from entities (per image)
        :param correct_labels: the hierarchy mapping
        :param url: the url field
        :return: objects_lst
        """

        # Initialized new objects list
        objects_lst = []
        for object in objects:

            # Get the label of object
            label = object.names[0]

            # Check if it is a correct label
            if label not in correct_labels:
                continue

            new_object_mapping = ObjectMapping(object.id, object.x, object.y, object.width, object.height, object.names,
                                               object.synsets, url)
            # Append the new objectMapping to objects_lst
            objects_lst.append(new_object_mapping)

        return objects_lst

    def draw_objects(self, entity, save_path=''):
        """
        This function draws the objects per entity
        :param entity: entity Visual Genome type
        :param save_path: the path to save the original gt scene graph
        :return:
        """
        if not save_path:
            print("No Outputs path has been receiving")
            return

        img = get_img(entity.image.url)
        index = 0

        # Store the positive and negative objects
        positive_objects = []
        negative_objects = []

        for object in entity.objects:
            try:
                # Get the mask: a dict with {x1,x2,y1,y2}
                mask_object = get_mask_from_object(object)
                # Saves as a box
                object_box = numpy.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
                object_label_predict = self.objects_index_labels[numpy.argmax(entity.objects_probs[index])]
                object_label_gt = object.names[0]

                # Define Subject color
                if object_label_gt != object_label_predict:
                    # Mistake
                    object_color = CvColor.BLACK
                    negative_objects.append(object)
                else:
                    # Correct
                    object_color = CvColor.GREEN
                    positive_objects.append(object)

                # Update mapping
                self.update_object_ids_to_obj_index(entity, object, index)

                # Draw Object box with their labels
                VisualizerDrawer.draw_labeled_box(img, object_box, label=object_label_predict + "/" + object_label_gt,
                                                  rect_color=object_color, scale=500)

                index += 1

            except Exception as e:
                print("Error: {}".format(str(e)))
                traceback.print_exc()

        path_save = os.path.join(save_path, "objects_img_{}.jpg".format(entity.image.id))
        cv2.imwrite(path_save, img)
        print("Objects image have been saved in {} \n".format(path_save))
        print("The Positive Objects are: {0} \n".format(positive_objects))
        print("The Negative Objects are: {0} \n".format(negative_objects))

    def update_object_ids_to_obj_index(self, entity, object, object_label_predict):
        """
        This function updates object_id and object_label prediction in object_ids_to_object_labels_mapping dict
        :param entity: entity
        :param object: object Visual Genome type
        :param object_label_predict: the prediction of the object label
        :return:
        """

        # Update mapping
        if entity.image.id not in self.object_ids_to_object_index_mapping:
            self.object_ids_to_object_index_mapping[entity.image.id] = {}

        self.object_ids_to_object_index_mapping[entity.image.id][object.id] = object_label_predict

    def draw_relationships(self, entity, save_path=''):
        """
        This function draws only the positive relationships (without "neg" predicate per entity)
        :param entity: entity Visual Genome type
        :param save_path: the path to save the original gt scene graph
        :return:
        """

        if not save_path:
            print("No Outputs path has been receiving")
            return

        # Check if this image id has already the object_ids to object_labels mapping
        if entity.image.id not in self.object_ids_to_object_index_mapping:
            self.find_objects_ids_to_object_labels_mapping(entity)

        # Get images
        img_relations_triplets = get_img(entity.image.url)
        img_positive_predicates = get_img(entity.image.url)
        img_relations_triplets_correct = get_img(entity.image.url)
        img_relations_triplets_mistake = get_img(entity.image.url)
        img_positive_predicates_correct = get_img(entity.image.url)
        img_positive_predicates_mistake = get_img(entity.image.url)
        index = 0

        # Get only positive relationships
        only_positive_relations = self.get_positive_relations(entity.relationships)
        num_of_iters = int(math.ceil(float(len(only_positive_relations)) / self.SPLIT_REL_PARAM))

        # Store the positive and negative predicates
        positive_relations = []
        negative_relations = []

        for batch_idx in range(num_of_iters):
            relations = only_positive_relations[self.SPLIT_REL_PARAM * batch_idx: self.SPLIT_REL_PARAM * (batch_idx + 1)]
            img_relations_triplets_batch_idx = get_img(entity.image.url)
            img_relations_pos_batch_idx = get_img(entity.image.url)

            for relation in relations:
                try:
                    predicate_gt = relation.predicate

                    # Object
                    # Get the mask: a dict with {x1,x2,y1,y2}
                    mask_object = get_mask_from_object(relation.object)
                    # Saves as a box
                    object_box = numpy.array(
                        [mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
                    object_label_gt = relation.object.names[0]
                    object_ind = self.object_ids_to_object_index_mapping[entity.image.id][relation.object.id]
                    object_label_predict = self.objects_index_labels[numpy.argmax(entity.objects_probs[object_ind])]

                    # Subject
                    # Get the mask: a dict with {x1,x2,y1,y2}
                    mask_subject = get_mask_from_object(relation.subject)
                    # Saves as a box
                    subject_box = numpy.array(
                        [mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])
                    subject_label_gt = relation.subject.names[0]
                    subject_ind = self.object_ids_to_object_index_mapping[entity.image.id][relation.subject.id]
                    subject_label_predict = self.objects_index_labels[numpy.argmax(entity.objects_probs[subject_ind])]

                    # Predicate
                    predicate_box = find_union_box(subject_box, object_box)
                    predicate_id_predict = numpy.argmax(entity.predicates_probes[subject_ind, object_ind])
                    predicate_label_predict = self.predicates_index_labels[predicate_id_predict]

                    # Define Subject color
                    if predicate_gt != predicate_label_predict:
                        predicate_color = CvColor.BLACK
                        negative_relations.append(relation)

                        ## Draw on the *WHOLE* *MISTAKE* relationships

                        # Draw Predicate box with their labels
                        VisualizerDrawer.draw_labeled_box(img_relations_triplets_mistake, predicate_box,
                                                          label="<{0}, {1}, {2}>/".format(subject_label_predict,
                                                                                          predicate_label_predict,
                                                                                          object_label_predict),
                                                          label2="<{0}, {1}, {2}>".format(subject_label_gt,
                                                                                          predicate_gt,
                                                                                          object_label_gt),
                                                          rect_color=predicate_color, scale=500)
                        # Draw Predicate box with their labels
                        VisualizerDrawer.draw_labeled_box(img_positive_predicates_mistake, predicate_box,
                                                          label=predicate_label_predict + "/" + predicate_gt,
                                                          rect_color=predicate_color, scale=500)

                    else:
                        predicate_color = CvColor.GREEN
                        positive_relations.append(relation)

                        ## Draw on the *WHOLE* *CORRECT* relationships

                        # Draw Predicate box with their labels
                        VisualizerDrawer.draw_labeled_box(img_relations_triplets_correct, predicate_box,
                                                          label="<{0}, {1}, {2}>/".format(subject_label_predict,
                                                                                          predicate_label_predict,
                                                                                          object_label_predict),
                                                          label2="<{0}, {1}, {2}>".format(subject_label_gt,
                                                                                          predicate_gt,
                                                                                          object_label_gt),
                                                          rect_color=predicate_color, scale=500)
                        # Draw Predicate box with their labels
                        VisualizerDrawer.draw_labeled_box(img_positive_predicates_correct, predicate_box,
                                                          label=predicate_label_predict + "/" + predicate_gt,
                                                          rect_color=predicate_color, scale=500)

                    ## Draw on *BATCH* relationships
                    # Draw Predicate box with their labels
                    VisualizerDrawer.draw_labeled_box(img_relations_triplets_batch_idx, predicate_box,
                                                      label="<{0}, {1}, {2}>/".format(subject_label_predict,
                                                                                      predicate_label_predict,
                                                                                      object_label_predict),
                                                      label2="<{0}, {1}, {2}>".format(subject_label_gt, predicate_gt,
                                                                                      object_label_gt),
                                                      rect_color=predicate_color, scale=500)

                    # Draw Predicate box with their labels
                    VisualizerDrawer.draw_labeled_box(img_relations_pos_batch_idx, predicate_box,
                                                      label=predicate_label_predict + "/" + predicate_gt,
                                                      rect_color=predicate_color, scale=500)

                    ## Draw on the *WHOLE* relationships
                    # Draw Predicate box with their labels
                    VisualizerDrawer.draw_labeled_box(img_relations_triplets, predicate_box,
                                                      label="<{0}, {1}, {2}>/".format(subject_label_predict,
                                                                                      predicate_label_predict,
                                                                                      object_label_predict),
                                                      label2="<{0}, {1}, {2}>".format(subject_label_gt, predicate_gt,
                                                                                      object_label_gt),
                                                      rect_color=predicate_color, scale=500)

                    # Draw Predicate box with their labels
                    VisualizerDrawer.draw_labeled_box(img_positive_predicates, predicate_box,
                                                      label=predicate_label_predict + "/" + predicate_gt,
                                                      rect_color=predicate_color, scale=500)

                    index += 1
                except Exception as e:
                    print("Error: {0} in iter {1}".format(str(e), index))
                    traceback.print_exc()

            path_save = os.path.join(save_path,
                                     "relations_triplets_batch{0}_from_{1}_to_{2}.jpg".format(batch_idx,
                                                                                              batch_idx * self.SPLIT_REL_PARAM,
                                                                                              (
                                                                                                  batch_idx + 1) * self.SPLIT_REL_PARAM))
            cv2.imwrite(path_save, img_relations_triplets_batch_idx)
            path_save = os.path.join(save_path,
                                     "positive_predicates_batch{0}_from_{1}_to_{2}.jpg".format(batch_idx,
                                                                                               batch_idx * self.SPLIT_REL_PARAM,
                                                                                               (
                                                                                                   batch_idx + 1) * self.SPLIT_REL_PARAM))
            cv2.imwrite(path_save, img_relations_pos_batch_idx)

        # SAVE the WHOLE images
        path_save = os.path.join(save_path, "relations_triplets_all.jpg")
        cv2.imwrite(path_save, img_relations_triplets)
        path_save = os.path.join(save_path, "positive_predicates_all.jpg")
        cv2.imwrite(path_save, img_positive_predicates)

        # SAVE the WHOLE CORRECT images
        path_save = os.path.join(save_path, "relations_correct_triplets.jpg")
        cv2.imwrite(path_save, img_relations_triplets_correct)
        path_save = os.path.join(save_path, "positive_correct_predicates.jpg")
        cv2.imwrite(path_save, img_positive_predicates_correct)

        # SAVE the WHOLE MISTAKE images
        path_save = os.path.join(save_path, "relations_mistake_triplets.jpg")
        cv2.imwrite(path_save, img_relations_triplets_mistake)
        path_save = os.path.join(save_path, "positive_mistake_predicates.jpg")
        cv2.imwrite(path_save, img_positive_predicates)
        cv2.imwrite(path_save, img_positive_predicates_mistake)

        print("Relationships image have been saved in {}".format(save_path))
        print("Predicates image have been saved in {} \n".format(save_path))

        # Print Positive and Negative relationships for STATS
        print("The Positive Relationships are: {0} \n".format(positive_relations))
        print("The Negative Relationships are: {0} \n".format(negative_relations))

    def draw_prediction_scene_graph(self, entity, save_path='', using_gt_object_boxes=False):
        """
        This function draws the scene graph (without "neg" predicate per entity)
        :param entity: entity Visual Genome type
        :param save_path: the path to save the original gt scene graph
        :param using_gt_object_boxes: A flag if we want to predict with GT object boxes or not.
        :return:
        """

        if not save_path:
            print("No Outputs path has been receiving")
            return

        # A numpy 1D array of indices of the predicted objects
        obj_arr = numpy.argmax(entity.objects_probs, axis=1)
        # A numpy 1D array of indices of the gt objects
        obj_gt_arr = numpy.argmax(entity.objects_labels, axis=1)
        # A numpy 2D array of indices of the gt objects
        pred_arr = numpy.argmax(entity.predicates_probes, axis=2)
        # A numpy 2D array of indices of the gt objects
        pred_gt_arr = numpy.argmax(entity.predicates_labels, axis=2)

        file_name = os.path.join(save_path, "{0}_predicted_scene_graph".format(entity.image.id))

        draw_graph(only_gt=False, pred=pred_arr, pred_gt=pred_gt_arr, obj=obj_arr, obj_gt=obj_gt_arr,
                   predicate_ids=self.predicates_index_labels, object_ids=self.objects_index_labels,
                   filename=file_name)

    def draw_original_gt_scene_graph(self, entity, save_path=''):
        """
        This function draws the original graph from scene graph (without "neg" predicate per entity).
        We need to create a new objects and predicates mapping for that.
        :param save_path: the path to save the original gt scene graph
        :param entity: entity Visual Genome type
        :return:
        """

        if not save_path:
            print("No Outputs path has been receiving")
            return

        # Create 3 mapping: objects_ids_to_indices, labels_to_index_objects and index_to_labels_objects
        objects_ids_to_indices = {}
        labels_to_index_objects = {}
        ind = 0
        unique_ind = 0
        for obj in entity.objects:

            # Update object_ids to object index mapping
            if obj.id not in objects_ids_to_indices:
                objects_ids_to_indices[obj.id] = ind

            # Update labels to index objects mapping
            if obj.names[0] not in labels_to_index_objects:
                labels_to_index_objects[obj.names[0]] = unique_ind
                unique_ind += 1

            ind += 1
        index_to_labels_objects = {ind: label for label, ind in labels_to_index_objects.iteritems()}

        # Create OBJECTS GT 1D array that has the object_ids inside
        objects_gt_arr = numpy.array([labels_to_index_objects[obj.names[0]] for obj in entity.objects])

        # Create 2 mapping: labels_to_index_predicates and index_to_labels_predicates
        labels_to_index_predicates = {}
        ind = 0
        for relation in entity.relationships:
            if relation.predicate not in labels_to_index_predicates:
                labels_to_index_predicates[relation.predicate] = ind
                ind += 1

        index_to_labels_predicates = {ind: label for label, ind in labels_to_index_predicates.iteritems()}

        # Create PREDICATES GT 2D array that has the predicates_ids inside
        pred_gt_arr = numpy.zeros((len(entity.objects), len(entity.objects)), dtype="int8")
        pred_gt_arr.fill(50)
        for relation in entity.relationships:
            object_id = relation.object.id
            object_ind = objects_ids_to_indices[object_id]
            subject_id = relation.subject.id
            subject_ind = objects_ids_to_indices[subject_id]
            predicate = relation.predicate
            pred_gt_arr[subject_ind][object_ind] = labels_to_index_predicates[predicate]

        # The file name which will be saved
        file_name = os.path.join(save_path, "{0}_original_scene_graph".format(entity.image.id))

        # Draw the graph
        draw_graph(only_gt=True, pred=None, pred_gt=pred_gt_arr, obj=None, obj_gt=objects_gt_arr,
                   predicate_ids=index_to_labels_predicates, object_ids=index_to_labels_objects,
                   filename=file_name)

    def find_objects_ids_to_object_labels_mapping(self, entity):
        """
        This function find the objects ids and their predicted label for relationship
        :param entity: entity Visual Genome type
        :return:
        """
        index = 0
        for object in entity.objects:
            object_label_predict = self.objects_index_labels[numpy.argmax(entity.objects_probs[index])]

            # Update mapping
            self.update_object_ids_to_obj_index(entity, object, index)

            index += 1

    def get_positive_relations(self, relationships):
        """
        This function returns only the positive relationships
        :param relationships: the entity.relationships (the whole including positive + negative)
        :return:
        """
        pos_rel = []
        for relation in relationships:
            if relation.predicate == 'neg':
                continue
            pos_rel.append(relation)

        return pos_rel

    def plot_stats(self, entity):
        """
        This function will plot an initialize statistics on the objects and relations
        :param entity:
        :return:
        """

        ## Stats for Objects
        objects = entity.objects
        # Get the GT labels - [len(objects), ]
        index_labels_per_gt_sample = numpy.array([self.hierarchy_mapping_objects[obj.names[0]] for obj in objects])
        # Get the max argument from the network output - [len(objects), ]
        index_labels_per_sample = numpy.argmax(entity.objects_probs, axis=1)

        print("The Total number of Objects is {0} and {1} of them are positives".format(
            len(objects), numpy.where(index_labels_per_gt_sample == index_labels_per_sample)[0].shape[0]))
        print("The Objects accuracy is {0}".format(
            numpy.where(index_labels_per_gt_sample == index_labels_per_sample)[0].shape[0] / float(len(objects))))

        ## Stats for Predicates
        # Get the labels of predicates
        predicates_labels = numpy.argmax(entity.predicates_labels, axis=2).reshape(-1)
        # Get the prediction of predicates
        predicates_prediction = numpy.argmax(entity.predicates_probes, axis=2).reshape(-1)
        # How many positive predicates we have
        pos_pred = numpy.sum(predicates_labels != self.hierarchy_mapping_predicates['neg'])
        # How many negative predicates we have
        neg_pred = numpy.sum(predicates_labels == self.hierarchy_mapping_predicates['neg'])

        print("The Total number of Relations is {0} while {1} of them positives and {2} of them negatives ".
              format(len(predicates_labels), pos_pred, neg_pred))

        # Number of the correct predictions
        correct_pred = numpy.sum(predicates_prediction == predicates_labels)
        print("The Total Relations accuracy is {0}".format(correct_pred / float(len(predicates_labels))))

        # Check for no divide by zero because we don't have any *POSITIVE* relations
        if pos_pred == 0:
            print("The Positive Relations accuracy is 0 - We have no positive relations")
        else:
            print("The Positive Relations accuracy is {0}".format(
                numpy.where((predicates_labels == predicates_prediction) &
                         (predicates_labels != self.hierarchy_mapping_predicates['neg']))[0].shape[0] /
                float(pos_pred)))

        # Check for no divide by zero because we don't have any *NEGATIVE* relations
        if neg_pred == 0:
            print("The Negative Relations accuracy is 0 - We have no negative relations")
        else:
            print("The Negative Relations accuracy is {0}".format(
                numpy.where((predicates_labels == predicates_prediction) &
                         (predicates_labels == self.hierarchy_mapping_predicates['neg']))[0].shape[0] /
                float(neg_pred)))
