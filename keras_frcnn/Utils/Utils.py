import numpy
import cv2
import math
import os

__author__ = 'roeih'

FILE_EXISTS_ERROR = (17, 'File exists')
VG_DATA_PATH = "Data/VisualGenome/data"
VG_PATCH_PATH = "Data/VisualGenome/Patches"
VG_PICKLES_FOLDER_PATH = "Data/VisualGenome/pickles"
VAL_IMGS_P = "val_imgs.p"
TRAIN_IMGS_P = "train_imgs.p"
CLASSES_COUNT_FILE = "classes_count.p"
CLASSES_MAPPING_FILE = "class_mapping.p"
RELATIONS_COUNT_FILE = "relations_count.p"
RELATIONS_MAPPING_FILE = "relations_mapping.p"
PREDICATES_COUNT_FILE = "predicates_count.p"
HIERARCHY_MAPPING = "hierarchy_mapping.p"
ENTITIES_FILE = "final_entities.p"
PascalVoc_PICKLES_PATH = "keras_frcnn/Data/PascalVoc"
VisualGenome_PICKLES_PATH = "keras_frcnn/Data/VisualGenome"
VG_VisualModule_PICKLES_PATH = "VisualModule/Data/VisualGenome"
MINI_VG_DATADET_PATH = "/home/roeih/VisualGenome/vg"
DATA_PATH = "Data/VisualGenome/data/"
TRAIN_DATA_SET = "train_set.p"
TEST_DATA_SET = "test_set.p"
VALIDATION_DATA_SET = "validation_set.p"
MINI_IMDB = "mini_imdb_1024.h5"



def convert_img_bgr_to_rgb(img):
    """
    This function convert image from BGR to RGB
    :param img_data: image Data
    :return: RGB img
    """
    # BGR -> RGB
    img = img[:, :, (2, 1, 0)]

    return img


def get_img_resize(image, image_width, image_height, type='pad'):
    if type == 'crop':
        msg = 'crop padding method is not supported'
        print(msg)
        raise AttributeError(msg)
        # return resizeImageCrop(image, image_width, image_height, crop=True)
    elif type == 'pad':
        # Not implemented
        return None
    elif type == 'zero_pad':
        return resize_image_zero_pad(image, image_width, image_height)
    elif type == 'avg_pad':
        # Not implemented
        return None
    else:
        return cv2.resize(image, (image_width, image_height))


def resize_image_zero_pad(image, image_width, image_height):
    """
    This function implements resize image with zero padding
    :param image: image
    :param image_width: image width
    :param image_height: image height
    :return: the image after resize with zero padding
    """
    if image is None:
        msg = 'Image cannot be None'
        print(msg)
        raise AttributeError(msg)

    # resize image for padding - keep the original sample AR
    # return: image with: one dimension=image_width/image_height
    # and one dimension < image_width/image_height which we have to pad
    image = resize_image_for_padding(image, image_width, image_height)
    if image_height > image.shape[0]:
        image = zero_pad_image(image, image_height - image.shape[0], axis=1)
    elif image_width > image.shape[1]:
        image = zero_pad_image(image, image_width - image.shape[1], axis=0)

    return image


def resize_image_for_padding(image, image_width, image_height):
    if image is None:
        msg = 'Image cannot be None'
        print(msg)
        raise AttributeError(msg)

    cur_image_height = image.shape[0]
    cur_image_width = image.shape[1]
    ratio = float(cur_image_height) / float(cur_image_width)

    height_scale = float(cur_image_height) / float(image_height)
    width_scale = float(cur_image_width) / float(image_width)

    if height_scale >= width_scale:
        new_width = int(math.floor(image_height / ratio))
        image = cv2.resize(image, (new_width, image_height))
    else:
        new_height = int(math.floor(image_width * ratio))
        image = cv2.resize(image, (image_width, new_height))

    return image


def zero_pad_image(image, size, axis=0):
    if axis:
        rows_to_pad_bottom = int(math.ceil(size / 2.))
        rows_to_pad_top = int(math.floor(size / 2.))

        if rows_to_pad_bottom >= 0 or rows_to_pad_top >= 0:
            image = numpy.lib.pad(image, ((rows_to_pad_top, rows_to_pad_bottom), (0, 0), (0, 0)), 'constant',
                                  constant_values=(0, 0))

    else:
        cols_to_pad_right = int(math.ceil(size / 2.))
        cols_to_pad_left = int(math.floor(size / 2.))
        if cols_to_pad_right >= 0 or cols_to_pad_left >= 0:
            image = numpy.lib.pad(image, ((0, 0), (cols_to_pad_left, cols_to_pad_right), (0, 0)), 'constant',
                                  constant_values=(0, 0))

    return image


def try_create_patch(image, mask, patch_path):
    try:
        # Cropping the patch from the image.
        patch = image[mask['y1']: mask['y2'], mask['x1']: mask['x2'], :]

        cv2.imwrite(patch_path, patch)

        # Try open the files to ensure they are valid
        image = cv2.imread(patch_path)

        if image is None:
            if os.path.exists(patch_path):
                os.remove(patch_path)
            return False
    except Exception as e:
        print(str(e))

        if os.path.exists(patch_path):
            os.remove(patch_path)

        return False

    return True


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
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


def get_mask_from_object(object):
    """
    This function gets object and returns its mask
    :param object: Object class
    :return: numpy dictionary as mask with x1,x2,y1,y2
    """
    x1 = object.x
    y1 = object.y
    width = object.width
    height = object.height
    x2 = x1 + width
    y2 = y1 + height
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
