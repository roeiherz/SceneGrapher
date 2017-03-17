import numpy as np
import cv2

__author__ = 'roeih'


def convert_img_bgr_to_rgb(img_data):
    """
    This function convert image from BGR to RGB
    :param img_data: image data
    :return: RGB img
    """
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    # img_data_aug = np.copy.deepcopy(img_data)
    img = cv2.imread(img_data['filepath'])
    # BGR -> RGB
    img = img[:, :, (2, 1, 0)]

    return img
