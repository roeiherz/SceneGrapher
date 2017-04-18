import numpy as np
import cv2

__author__ = 'roeih'


def augment_pascal_voc(img, img_data, config):
    """
    This function augment the data according to the config file
    :param img: an image numpy type
    :param img_data: image data
    :param config: a config class
    :return:
    """
    img_data_aug = np.copy(img_data)
    rows, cols = img.shape[:2]

    if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        for bbox in img_data_aug['bboxes']:
            x1 = bbox['x1']
            x2 = bbox['x2']
            bbox['x2'] = cols - x1
            bbox['x1'] = cols - x2

    if config.use_vertical_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 0)
        for bbox in img_data_aug['bboxes']:
            y1 = bbox['y1']
            y2 = bbox['y2']
            bbox['y2'] = rows - y1
            bbox['y1'] = rows - y2

    if config.random_rotate:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                    np.random.randint(-config.random_rotate_scale,
                                                      config.random_rotate_scale), 1)
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


def augment_visual_genome(patch, object, config, mask):
    """
    This function augment the data according to the config file
    :param patch: an image numpy type
    :param object: object Visual Genome class
    :param config: a config class
    :param mask: a dict with {x1,x2,y1,y2} for each object
    :return:
    """
    rows, cols = patch.shape[:2]

    # new_patches = []
    # new_labels = []

    if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
        new_patch = cv2.flip(patch, 1)
        # new_patches.append(new_patch)
        # new_labels.append(object.names[0])

    if config.use_vertical_flips and np.random.randint(0, 2) == 0:
        new_patch = cv2.flip(patch, 0)
        # new_patches.append(new_patch)
        # new_labels.append(object.names[0])

    if config.random_rotate:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                    np.random.randint(-config.random_rotate_scale,
                                                      config.random_rotate_scale), 1)
        new_patch = cv2.warpAffine(patch, M, (cols, rows), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

        # new_patches.append(new_patch)
        # new_labels.append(object.names[0])

        # K = np.array([[mask['x1'], mask['y1']], [mask['x2'], mask['y2']], [mask['x1'], mask['y2']],
        #               [mask['x2'], mask['y1']]])
        # K = cv2.transform(K.reshape(4, 1, 2), M)[:, 0, :]

        # (x1, y1) = np.min(K, axis=0)
        # (x2, y2) = np.max(K, axis=0)
        #
        # mask['x1'] = x1
        # mask['x2'] = x2
        # mask['y1'] = y1
        # mask['y2'] = y2

    return new_patch
