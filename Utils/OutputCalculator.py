import abc
import cPickle
from operator import itemgetter
import numpy as np
import cv2
import os
from Utils import create_folder
from FeaturesExtraction.Utils.Boxes import find_union_box, iou
from FeaturesExtraction.Utils.Utils import get_img, get_mask_from_object
from FeaturesExtraction.Utils.Visualizer import VisualizerDrawer, CvColor
import pandas as pd
from DesignPatterns.DetectionsStats import DetectionsStats
import time

__author__ = 'roeih'

DRAWS_FOLDER = "Draws"
STATS_FOLDER = "Stats"
STATS_SUMMARY_FILE_NAME = "summary"
CSV_END_FILE = ".csv"


class OutputCalculator(object):
    @abc.abstractmethod
    def collect(self):
        pass

    @abc.abstractmethod
    def export(self, draw=False, collect=False):
        pass

    @abc.abstractmethod
    def draw(self, detection, img, path_file, img_id):
        pass


class RelationCalculator(OutputCalculator):
    def __init__(self, file_path):
        self._file_path = file_path
        self._stats = None
        self._summary_df = None
        self._entities_id_df = None

    def collect(self):
        """
        This function collect the stats file and initialize the stats data frames
        """
        if self._stats is None:
            self._stats = cPickle.load(open(self._file_path))

        if self._summary_df is None:
            self._summary_df = pd.DataFrame(columns=("Images_Ids", "Predicate_Ids", "Predicate_Accuracy", "Subject_Ids",
                                                     "Subject_Accuracy", "Object_Ids", "Object_Accuracy",
                                                     "Relation_Ids", "Relation_Accuracy"))

        if self._entities_id_df is None:
            self._entities_id_df = pd.DataFrame()

    @staticmethod
    def predicates_query(detections_stats):
        """
        This function check the detections that have a correct prediction of predicate
        :param detections_stats: a DetectionsStats numpy dtype which is sorted
        :return: the predicates_ids (numpy of TOP_K indices) and predicates_acc (float)
        """
        predicates = np.where(
            detections_stats[DetectionsStats.PredictPredicate] == detections_stats[DetectionsStats.Predicate])
        predicates_ids = predicates[0]
        predicate_acc = float(len(predicates[0])) / len(detections_stats)
        return predicates_ids, predicate_acc

    @staticmethod
    def relations_query(detections_stats):
        """
        This function check the detections that have a correct full relation <subject, predicate, object>
        :param detections_stats: a DetectionsStats numpy dtype which is sorted
        :return: the relation_ids (numpy of TOP_K indices) and predicates_acc (float)
        """
        relations = np.where(
            (detections_stats[DetectionsStats.PredictPredicate] == detections_stats[DetectionsStats.Predicate]) &
            (detections_stats[DetectionsStats.PredictSubjectClassifications] ==
             detections_stats[DetectionsStats.SubjectClassifications]) &
            (detections_stats[DetectionsStats.PredictObjectClassifications] ==
             detections_stats[DetectionsStats.ObjectClassifications]))

        relations_ids = relations[0]
        relations_acc = float(len(relations[0])) / len(detections_stats)
        return relations_ids, relations_acc

    @staticmethod
    def subjects_query(detections_stats):
        """
        This function check the detections that have a correct prediction of subject
        :param detections_stats: a DetectionsStats numpy dtype which is sorted
        :return: the subjects_ids (numpy of TOP_K indices) and subjects_acc (float)
        """
        subjects = np.where(detections_stats[DetectionsStats.PredictSubjectClassifications] == detections_stats[
            DetectionsStats.SubjectClassifications])

        subjects_ids = subjects[0]
        subjects_acc = float(len(subjects[0])) / len(detections_stats)
        return subjects_ids, subjects_acc

    @staticmethod
    def objects_query(detections_stats):
        """
        This function check the detections that have a correct prediction of objects
        :param detections_stats: a DetectionsStats numpy dtype which is sorted
        :return: the objects_ids (numpy of TOP_K indices) and objects_acc (float)
        """
        objects = np.where(detections_stats[DetectionsStats.PredictObjectClassifications] == detections_stats[
            DetectionsStats.ObjectClassifications])

        objects_ids = objects[0]
        objects_acc = float(len(objects[0])) / len(detections_stats)
        return objects_ids, objects_acc

    def export(self, draw=False, stats=False):
        """

        :return:
        """

        # Sorted by entity confidence
        stats_sorted = sorted(self._stats, key=itemgetter(2), reverse=True)
        ind = 0
        # Time stamp
        time_stamp = time.strftime("%d_%m_%Y-%H_%M_%S")

        for stats_tuple in stats_sorted:
            detections_stats = stats_tuple[0]
            entity = stats_tuple[1]
            entity_confidence = stats_tuple[2]

            url = entity.image.url
            img_id = entity.image.id
            iou(stats_tuple, stats_tuple)
            # Calculates Stats
            if stats:
                # Calculates predicates ids and acc (if predicate is the same)
                predicates_ids, predicate_acc = self.predicates_query(detections_stats)

                # Calculates relations ids and acc (if we have a correct full relation <subject, predicate, object>)
                relation_ids, relations_acc = self.relations_query(detections_stats)

                # Calculates subjects ids and acc (if predicate is the same)
                subjects_ids, subjects_acc = self.subjects_query(detections_stats)

                # Calculates subjects ids and acc (if predicate is the same)
                objects_ids, objects_acc = self.objects_query(detections_stats)

                # Add row to the summary data frame
                self._summary_df.loc[ind] = [int(img_id), predicates_ids, predicate_acc, subjects_ids, subjects_acc,
                                             objects_ids, objects_acc, relation_ids, relations_acc]

            # Draw detections which are sorted by their entity confidence
            if draw:

                # Create the path for draws
                path_draws = os.path.join(DRAWS_FOLDER, time_stamp, str(img_id))

                # Entity confidence is
                if entity_confidence == 0.0:
                    path_draws += "_neg"
                else:
                    path_draws += "_pos"

                # Create folder if its not exists
                create_folder(path_draws)

                # Draw DetectionsStats
                self.draw(detections_stats, url, path_draws, img_id)

            # Increment index
            ind += 1

        # Create the path for draws
        path_stats = os.path.join(STATS_FOLDER, time_stamp)
        # Create folder if its not exists
        create_folder(path_stats)
        # Stats file name
        stats_file = "{0}{1}".format(STATS_SUMMARY_FILE_NAME, CSV_END_FILE)
        # Save the stats file
        self._summary_df.to_csv(os.path.join(path_stats, stats_file))

        print("END")

    def draw(self, detections_stats, url, path_file, img_id):
        """
        This function draws detections in image and save it in path file
        :param img_id: image id (entity id)
        :param detections_stats: a DetectionsStats numpy dtype which are sorted by their entity confidence
        :param url: the url of the entity
        :param path_file: the path
        :return:
        """

        for detection in detections_stats:

            # Get the image
            img = get_img(url, download=True)

            if img is None:
                print("Print Image Is None with url: {}".format(url))
                continue

            # Get the Subject Box
            subject_box = detection[DetectionsStats.SubjectBox]

            # Get the Object Box
            object_box = detection[DetectionsStats.ObjectBox]

            # Draw Subject
            VisualizerDrawer.draw_labeled_box(img, subject_box,
                                              label=detection[DetectionsStats.PredictSubjectClassifications] + "/" +
                                                    detection[DetectionsStats.SubjectClassifications],
                                              rect_color=CvColor.BLUE, scale=2000, where="top_left")

            # Draw Object
            VisualizerDrawer.draw_labeled_box(img, object_box,
                                              label=detection[DetectionsStats.PredictObjectClassifications] + "/" +
                                                    detection[DetectionsStats.ObjectClassifications],
                                              rect_color=CvColor.BLACK, scale=2000, where="top_left")

            # Draw Union-Box Predicate
            VisualizerDrawer.draw_labeled_box(img, find_union_box(subject_box, object_box),
                                              label=detection[DetectionsStats.PredictPredicate] + "/" +
                                                    detection[DetectionsStats.Predicate],
                                              rect_color=CvColor.GREEN, scale=2000, where="center")

            cv2.imwrite(os.path.join(path_file, "{0}_<{1}_{2}_{3}>\<{4}_{5}_{6}>_{7}.jpg".format(img_id, detection[
                DetectionsStats.PredictSubjectClassifications],
                                                                                                 detection[
                                                                                                     DetectionsStats.PredictPredicate],
                                                                                                 detection[
                                                                                                     DetectionsStats.PredictObjectClassifications],
                                                                                                 detection[
                                                                                                     DetectionsStats.SubjectClassifications],
                                                                                                 detection[
                                                                                                     DetectionsStats.Predicate],
                                                                                                 detection[
                                                                                                     DetectionsStats.ObjectClassifications],
                                                                                                 detection[
                                                                                                     DetectionsStats.TopKIndex])),
                        img)


if __name__ == '__main__':
    print("Start Relation Calculator")
    rl = RelationCalculator(file_path="detections_stat.p")
    rl.collect()

    stats_sorted = sorted(rl._stats, key=itemgetter(2), reverse=True)
    tt = [stats_tuple if stats_tuple[1].image.id == 2338752 else None for stats_tuple in stats_sorted]
    cc = []
    for t in tt:
        if t:
            cc.append(t)
    aa = cc[0]
    aa_objects = [aa[1].objects[i] for i in range(len(aa[1].objects))]
    aa_masks = [get_mask_from_object(object) for object in aa_objects]
    aa_boxes = [np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']]) for mask_object
                in aa_masks]
    box_aa = aa[0][41][DetectionsStats.SubjectBox]
    dd = np.where(np.in1d(list(aa[0][DetectionsStats.TopKIndex]), [12, 41, 53, 64, 77, 82, 83, 51]) == True)[0]
    ind = np.array([12, 41, 53, 64, 77, 82, 83, 51])
    bb = aa[0][dd]
    rl.export(draw=False, stats=True)
    print('End')
