import os
import xml.etree.ElementTree as ET

import cv2

from features_extraction.Lib.DataSetGenerator import DataSetGenerator

__author__ = 'roeih'


class PascalVoc(DataSetGenerator):
    """
    This class represents PascalVoc wrapper
    """

    def __init__(self, name='pascal_voc', visualize=False):
        super(PascalVoc, self).__init__(name, visualize)

    def get_data(self, input_path, pascal_data=['VOC2007', 'VOC2012']):
        """
        This function will store PascalVoc Data
        :param pascal_data:
        :param input_path: path for input
        :return:
        """

        all_imgs = []
        classes_count = {}
        class_mapping = {}
        data_paths = [os.path.join(input_path, s) for s in pascal_data]

        print('Parsing annotation files')

        for data_path in data_paths:

            annot_path = os.path.join(data_path, 'Annotations')
            imgs_path = os.path.join(data_path, 'JPEGImages')
            imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
            imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

            trainval_files = []
            test_files = []
            try:
                with open(imgsets_path_trainval) as f:
                    for line in f:
                        trainval_files.append(line.strip() + '.jpg')
                with open(imgsets_path_test) as f:
                    for line in f:
                        test_files.append(line.strip() + '.jpg')
            except Exception as e:
                print(e)

            annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
            idx = 0
            for annot in annots:
                try:
                    idx += 1

                    et = ET.parse(annot)
                    element = et.getroot()

                    element_objs = element.findall('object')
                    element_filename = element.find('filename').text
                    element_width = int(element.find('size').find('width').text)
                    element_height = int(element.find('size').find('height').text)

                    if len(element_objs) > 0:
                        annotation_data = {'filepath': os.path.join(imgs_path, element_filename),
                                           'width': element_width,
                                           'height': element_height, 'bboxes': []}
                        if element_filename in trainval_files:
                            annotation_data['imageset'] = 'trainval'
                        elif element_filename in test_files:
                            annotation_data['imageset'] = 'test'
                        else:
                            annotation_data['imageset'] = 'test'

                    for element_obj in element_objs:
                        class_name = element_obj.find('name').text
                        if class_name not in classes_count:
                            classes_count[class_name] = 1
                        else:
                            classes_count[class_name] += 1

                        if class_name not in class_mapping:
                            class_mapping[class_name] = len(class_mapping)

                        obj_bbox = element_obj.find('bndbox')
                        x1 = int(round(float(obj_bbox.find('xmin').text)))
                        y1 = int(round(float(obj_bbox.find('ymin').text)))
                        x2 = int(round(float(obj_bbox.find('xmax').text)))
                        y2 = int(round(float(obj_bbox.find('ymax').text)))
                        annotation_data['bboxes'].append(
                            {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})

                    all_imgs.append(annotation_data)

                    if self._visualize:
                        img = cv2.imread(annotation_data['filepath'])
                        for bbox in annotation_data['bboxes']:
                            cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
                                                                              'x2'], bbox['y2']), (0, 0, 255))
                        cv2.imshow('img', img)
                        cv2.waitKey(0)

                except Exception as e:
                    print(e)
                    continue
        return all_imgs, classes_count, class_mapping
