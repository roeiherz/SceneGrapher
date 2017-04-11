import numpy as np
import cv2

__author__ = 'roeih'


class DataAugmention(object):
    """
    This class is a Data augmentation
    """

    def __init__(self, img, img_data, config):
        self._img = img
        self._config = config
        self._img_data = img_data
        # self._img_data = self._get_img_data_for_VG(img_data) if self._config.dataset == "VisualGenome" else img_data

    def _get_img_data_for_VG(self, img_data):
        """
        
        :param img_data: 
        :return: 
        """


    def augment(self):
        """
        This function augment the Data according to the config file
        :return:
        """
        img_data_aug = np.copy(self._img_data)
        img = self._img
        rows, cols = img.shape[:2]

        if self._config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if self._config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if self._config.random_rotate:
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                        np.random.randint(-self._config.random_rotate_scale,
                                                          self._config.random_rotate_scale), 1)
            img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

            for bbox in img_data_aug['bboxes']:
                K = np.array([[bbox['x1'], bbox['y1']], [bbox['x2'], bbox['y2']], [bbox['x1'], bbox['y2']],
                              [bbox['x2'], bbox['y1']]])
                K = cv2.transform(K.reshape(4, 1, 2), M)[:, 0, :]

                (x1, y1) = np.min(K, axis=0)
                (x2, y2) = np.max(K, axis=0)

                bbox['x1'] = x1
                bbox['x2'] = x2
                bbox['y1'] = y1
                bbox['y2'] = y2

        return img_data_aug, img
