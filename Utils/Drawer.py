from FeaturesExtraction.Utils.Utils import get_img, PROJECT_ROOT, get_mask_from_object
from FeaturesExtraction.Utils.Visualizer import VisualizerDrawer, CvColor
from FilesManager.FilesManager import FilesManager
from Logger import Logger
import cPickle
import os
import numpy as np
import cv2

__author__ = 'roeih'


def entity_drawer(entity, draw_objects=True, draw_relations=False):
    """
    This function draws an entity in image and save it in path file
    :param draw_relations: A flag if we are drawing relations
    :param draw_objects: A flag if we are drawing objects
    :param entity: An entity 
    :return: 
    """

    # Get the image
    img = get_img(entity.image.url, download=True)

    if img is None:
        Logger().log("Print Image Is None with url: {}".format(entity.image.url))
        return

    if draw_objects:
        for object in entity.objects:
            # Get the mask: a dict with {x1,x2,y1,y2}
            mask_object = get_mask_from_object(object)
            # Saves as a box
            object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
            # Draw Object
            VisualizerDrawer.draw_labeled_box(img, object_box,
                                              label=object.names[0],
                                              rect_color=CvColor.BLUE, scale=500, where="top_left")

    output_dir = os.path.join(PROJECT_ROOT, "Pics")
    output_file_name = "{0}.jpg".format(entity.image.id)
    cv2.imwrite(os.path.join(output_dir, output_file_name), img)


if __name__ == '__main__':

    # Define FileManager
    filemanager = FilesManager()
    # Define Logger
    logger = Logger()

    # Load the entities from a file
    file_path = os.path.join(PROJECT_ROOT,
                             "FilesManager/FeaturesExtraction/PredicatedFeatures/Wed_Aug__9_10:04:43_2017/test.p")
    entities = cPickle.load(open(file_path))
    for entity in entities:
        entity_drawer(entity)

