import abc
import cPickle
from operator import itemgetter
import numpy as np
import cv2
import os
from Utils import create_folder
from keras_frcnn.Utils.Boxes import find_union_box
from keras_frcnn.Utils.Utils import get_img
from keras_frcnn.Utils.Visualizer import VisualizerDrawer, CvColor

from DesignPatterns.DetectionsStats import DetectionsStats

__author__ = 'roeih'

PICS_FOLDER = "Pics"


class OutputCalculator(object):
    @abc.abstractmethod
    def collect(self):
        pass

    @abc.abstractmethod
    def export(self):
        pass

    @abc.abstractmethod
    def draw(self):
        pass


class RelationCalculator(OutputCalculator):
    def __init__(self, file_path):
        self._file_path = file_path
        self._stats = None

    def collect(self):
        """
        This function collect the stats file
        """
        if self._stats is None:
            self._stats = cPickle.load(open(self._file_path))

    def export(self):
        """

        :return:
        """

        # Sorted by entity confidence
        stats_sorted = sorted(self._stats, key=itemgetter(2), reverse=True)

        for stats_tuple in stats_sorted:
            detections_stats = stats_tuple[0]
            entity_stats = stats_tuple[1]
            entity_confidence = stats_tuple[2]

            url = entity_stats.image.url
            id = entity_stats.image.id
            path_file = os.path.join(PICS_FOLDER, str(id))
            create_folder(path_file)

            np.where(detections_stats[DetectionsStats.Url])

            # The detections are been sorted
            for detection in detections_stats:
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

                # Draw Subject
                VisualizerDrawer.draw_labeled_box(img, object_box,
                                                  label=detection[DetectionsStats.PredictObjectClassifications] + "/" +
                                                        detection[DetectionsStats.ObjectClassifications],
                                                  rect_color=CvColor.BLACK, scale=2000, where="top_left")

                # Draw Union-Box Predicate
                VisualizerDrawer.draw_labeled_box(img, find_union_box(subject_box, object_box),
                                                  label=detection[DetectionsStats.PredictPredicate] + "/" +
                                                        detection[DetectionsStats.Predicate],
                                                  rect_color=CvColor.GREEN, scale=2000, where="center")

                cv2.imwrite(os.path.join(path_file, "{0}_<{1}_{2}_{3}>\<{4}_{5}_{6}>_{7}.jpg".format(id,
                                                                                                     detection[
                                                                                                         DetectionsStats.SubjectClassifications],
                                                                                                     detection[
                                                                                                         DetectionsStats.Predicate],
                                                                                                     detection[
                                                                                                         DetectionsStats.ObjectClassifications],
                                                                                                     detection[
                                                                                                         DetectionsStats.PredictSubjectClassifications],
                                                                                                     detection[
                                                                                                         DetectionsStats.PredictPredicate],
                                                                                                     detection[
                                                                                                         DetectionsStats.PredictObjectClassifications],
                                                                                                     detection[
                                                                                                         DetectionsStats.TopKIndex])),
                            img)

            # Entity confidence is
            if entity_confidence == 0.0:
                break

        print("Hi")

    def draw(self):
        pass


if __name__ == '__main__':
    print("Start Relation Calculator")
    tt = RelationCalculator(file_path="detections_stat.p")
    tt.collect()
    tt.export()
    print('End')
