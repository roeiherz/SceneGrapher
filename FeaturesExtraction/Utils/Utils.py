import math
import os
import threading
import time
import urllib
import cv2
import matplotlib.pyplot as plt
import numpy
from keras.engine import Model
from keras.layers import Dense

# todo: when moving to nova remove it
from FilesManager.FilesManager import FilesManager

# PROJECT_ROOT = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/"
PROJECT_ROOT = "/home/roeih/SceneGrapher/"
VG_DATA_PATH_OLD = "Data/VisualGenome/data"
VG_DATA_PATH = "Data/VisualGenome/preprocessed_data"
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
PascalVoc_PICKLES_PATH = "FeaturesExtraction/Data/PascalVoc"
VisualGenome_PICKLES_PATH = "FeaturesExtraction/Data/VisualGenome"
VisualGenome_DATASETS_PICKLES_PATH = "FeaturesExtraction/PicklesDataset"
VG_VisualModule_PICKLES_PATH = "VisualModule/Data/VisualGenome"
MINI_VG_DATASET_PATH = "Data/VisualGenome/mini"
OBJECTS_ALIAS = "object_alias.txt"
OBJECTS_LIST = "object_list.txt"
PREDICATES_ALIAS = "predicate_alias.txt"
PREDICATES_LIST = "predicate_list.txt"
DATA_PATH_OLD = "Data/VisualGenome/data/"
DATA_PATH = "Data/VisualGenome/preprocessed_data/"
TRAIN_DATA_SET = "train_set.p"
TEST_DATA_SET = "test_set.p"
VALIDATION_DATA_SET = "validation_set.p"
MINI_IMDB = "mini_imdb_1024.h5"
TRAINING_OBJECTS_CNN_PATH = "FilesManager/FeaturesExtraction/ObjectsCNN"
TRAINING_PREDICATE_CNN_PATH = "FilesManager/FeaturesExtraction/PredicatesCNN"
TRAINING_PREDICATE_MASK_CNN_PATH = "FilesManager/FeaturesExtraction/PredicatesMaskCNN"
PREDICATED_FEATURES_PATH = "FilesManager/FeaturesExtraction/PredicatedFeatures"
OUTPUTS_PATH = "Outputs"
FILTERED_DATA_SPLIT_PATH = "FilesManager/Data/VisualGenome/FilteredData"
PREPROCESSED_FILTERED_DATA_SPLIT_PATH = "FilesManager/Data/VisualGenome/PreProcessedFilteredData"
EXTRACTED_DATA_SPLIT_PATH = "FilesManager/Data/VisualGenome/ExtractedData"
WEIGHTS_NAME = 'model_vg_resnet50.hdf5'
DATA = "data"
VISUAL_GENOME = "visual_genome"
POSITIVE_NEGATIVE_RATIO = 3


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


def get_time_and_date():
    """
    This function returns the time and the date without spaces mainly for saving files
    :return: string file which contains time and the date without spaces
    """
    return time.strftime("%c").replace(" ", "_")


def plot_graph(folder_path=""):
    """
    This function creates the error, accuracy and loss graph for training and testing and saved them to a folder path
    :param folder_path: the path for the training log folder (not file)
    """

    training_log_file = os.path.join(folder_path, "training.log")

    if not os.path.exists(training_log_file):
        print("Error with the file path. The file is not exist")

    # Open the file
    log = open(training_log_file, 'rb')
    first_line = True

    # Parser the file
    for line in log:
        try:
            # Remove the \r\n from the line
            line_splits = line.splitlines()
            # Split the line with the delimiter ",
            data_line = line_splits[0].split(",")

            if first_line:
                # The first line is creating the data dict with the keys from the first line
                data_dict = {key: [] for key in data_line}
                first_line = False
                # Continue to the next line
                continue

            # Fill the dict
            data_dict['epoch'].append(int(data_line[0]))
            data_dict['acc'].append(float(data_line[1]))
            data_dict['loss'].append(float(data_line[2]))
            data_dict['val_acc'].append(float(data_line[3]))
            data_dict['val_loss'].append(float(data_line[4]))

        except Exception as e:
            print("Exception while parser training.log")
            print(str(e))

    # Graph for model accuracy
    plt.figure()
    plt.plot(data_dict['acc'])
    plt.plot(data_dict['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folder_path, "model_accuracy.jpg"))
    plt.close()

    # Graph for model error
    plt.figure()
    plt.plot([1 - acc for acc in data_dict['acc']])
    plt.plot([1 - acc for acc in data_dict['val_acc']])
    plt.title('model error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folder_path, "model_error.jpg"))
    plt.close()

    # Graph for model loss
    plt.figure()
    plt.plot(data_dict['loss'])
    plt.plot(data_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folder_path, "model_loss.jpg"))
    plt.close()


def replace_top_layer(model, num_of_classes):
    """
    This function replaces the last top layer (Dense layer) in a new layer
    :param num_of_classes: number of new classes for the new Dense layer
    :param model: the model
    :return: the updated model
    """
    # Remove the Dense layer and replace it with another
    model.layers.pop()
    # Define new layer
    new_output_layer = Dense(num_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
        model.layers[-1].output)
    # Define the new model
    model = Model(inputs=model.input, outputs=new_output_layer, name='resnet50')
    return model


def get_img(url, download=False, original=False):
    """
    This function read image from VisualGenome dataset as url and returns the image from local hard-driver
    :param original: Do you want to use original images (not pre-processed)
    :param download: A flag if we want to download the image
    :param url: url of the image
    :return: the image
    """
    try:
        path_lst = url.split('/')
        img_path = os.path.join(PROJECT_ROOT, VG_DATA_PATH, path_lst[-2], path_lst[-1])
        if original:
            img_path = os.path.join(PROJECT_ROOT, VG_DATA_PATH_OLD, path_lst[-2], path_lst[-1])

        if not os.path.isfile(img_path):
            print("Error. Image path was not found")
            # Download the image
            if download:
                downloadProbe(img_path, url)

        img = cv2.imread(img_path)

    except Exception as e:
        print(str(e))
        return None

    return img


def get_sorting_url():
    """
    This function sorting urls
    :return: a list of urls
    """
    bad_lst = get_bad_urls()

    mini_entities_lst = [u'https://cs.stanford.edu/people/rak248/VG_100K/2339172.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339171.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339170.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339169.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339168.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339167.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339166.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339165.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339164.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339163.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339162.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339161.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339160.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339159.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339158.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339157.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339156.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339155.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339154.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339153.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339152.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339151.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339150.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339149.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339148.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339147.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339146.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339145.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339144.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339143.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339142.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339141.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339140.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339139.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339138.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339137.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339136.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339135.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339134.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339133.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339132.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339131.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339130.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339129.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339127.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339126.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339125.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339124.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339123.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339122.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339121.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339120.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339119.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339118.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339117.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339116.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339115.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339114.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339113.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339112.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339110.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339109.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339108.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339107.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339106.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339105.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339104.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339103.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339102.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339101.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339100.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339099.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339098.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339097.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339096.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339095.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339094.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339093.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339092.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339091.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339090.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339089.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339088.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339087.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339086.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339085.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339084.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339083.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339082.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339081.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339080.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339079.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339078.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339077.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339076.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339075.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339074.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339073.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339072.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339071.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339070.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339069.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339068.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339067.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339066.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339065.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339064.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339063.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339062.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339061.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339060.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339059.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339058.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339057.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339056.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339055.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339054.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339053.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339052.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339051.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339050.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339049.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339048.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339047.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339046.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339045.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339044.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339043.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339042.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339041.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339040.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339039.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339037.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339036.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339034.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339033.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339031.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339030.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339029.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339028.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339027.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339026.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339025.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339024.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339023.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339022.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339021.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339020.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339019.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339018.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339017.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339016.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339015.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339014.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339013.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339012.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339011.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339010.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339009.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339008.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339007.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339006.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339005.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339004.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339003.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339002.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339001.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2339000.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338999.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338998.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338997.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338996.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338995.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338994.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338993.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338992.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338991.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338990.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338989.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338988.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338987.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338986.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338985.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338984.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338983.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338982.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338981.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338980.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338979.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338978.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338977.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338976.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338975.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338974.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338973.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338972.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338971.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338970.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338969.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338968.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338967.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338966.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338965.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338964.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338963.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338962.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338961.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338960.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338959.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338958.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338957.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338956.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338955.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338954.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338953.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338952.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338951.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338950.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338949.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338948.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338947.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338946.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338945.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338944.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338943.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338942.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338941.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338940.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338939.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338938.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338937.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338936.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338935.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338934.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338933.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338932.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338931.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338930.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338929.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338928.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338927.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338925.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338924.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338923.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338922.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338921.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338920.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338919.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338918.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338917.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338916.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338915.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338914.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338913.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338912.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338911.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338910.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338909.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338908.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338907.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338906.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338905.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338904.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338903.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338902.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338901.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338900.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338899.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338898.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338897.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338896.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338895.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338894.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338893.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338891.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338890.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338889.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338888.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338887.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338886.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338885.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338882.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338881.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338880.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338879.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338878.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338877.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338876.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338875.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338874.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338873.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338872.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338871.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338870.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338869.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338868.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338867.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338866.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338865.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338864.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338863.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338862.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338861.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338860.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338859.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338858.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338857.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338856.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338855.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338854.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338853.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338852.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338851.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338850.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338849.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338848.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338847.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338846.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338845.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338844.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338843.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338842.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338841.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338840.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338839.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338838.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338837.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338836.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338835.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338834.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338833.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338831.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338830.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338829.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338828.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338827.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338826.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338825.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338824.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338822.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338821.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338820.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338819.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338817.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338816.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338815.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338814.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338812.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338811.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338810.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338809.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338808.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338807.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338806.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338805.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338804.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338803.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338802.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338801.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338799.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338798.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338797.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338796.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338795.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338794.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338793.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338792.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338790.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338789.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338788.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338787.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338786.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338785.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338784.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338783.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338782.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338781.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338780.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338779.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338778.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338777.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338776.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338775.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338774.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338773.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338772.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338771.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338770.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338768.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338767.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338766.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338765.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338763.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338762.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338761.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338760.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338759.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338757.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338756.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338755.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338754.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338753.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338752.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338751.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338750.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338749.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338748.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338746.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338745.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338744.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338743.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338742.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338740.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338739.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338738.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338737.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338736.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338735.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338734.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338733.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338732.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338731.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338730.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338729.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338728.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338727.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338726.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338725.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338724.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338723.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338722.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338721.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338720.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338719.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338718.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338717.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338716.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338715.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338714.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338713.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338712.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338711.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338710.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338709.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338708.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338707.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338706.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338705.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338704.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338703.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338702.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338701.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338700.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338699.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338698.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338697.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338696.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338695.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338694.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338693.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338692.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338691.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338690.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338689.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338688.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338687.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338686.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338685.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338684.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338683.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338682.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338681.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338680.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338679.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338678.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338677.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338676.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338675.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338674.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338673.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338672.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338671.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338670.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338669.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338668.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338666.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338665.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338664.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338663.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338662.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338661.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338660.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338659.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338658.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338657.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338656.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338655.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338653.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338652.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338651.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338650.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338649.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338648.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338647.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338646.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338645.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338644.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338643.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338642.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338641.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338640.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338639.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338638.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338637.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338636.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338634.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338633.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338632.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338631.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338630.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338629.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338628.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338627.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338626.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338625.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338624.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338621.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338620.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338619.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338618.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338617.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338616.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338615.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338614.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338613.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338612.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338611.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338610.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338609.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338608.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338607.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338606.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338605.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338603.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338602.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338601.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338600.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338599.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338598.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338597.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338596.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338595.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338594.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338593.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338592.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338591.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338590.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338589.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338588.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338587.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338586.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338585.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338584.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338583.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338582.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338581.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338580.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338578.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338577.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338575.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338574.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338573.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338572.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338571.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338570.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338569.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338568.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338567.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338566.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338565.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338564.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338563.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338562.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338561.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338560.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338559.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338558.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338557.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338556.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338555.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338554.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338553.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338552.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338551.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338550.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338549.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338548.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338547.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338546.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338545.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338544.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338543.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338542.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338541.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338540.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338539.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338538.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338537.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338536.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338535.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338534.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338533.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338532.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338531.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338530.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338529.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338528.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338527.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338526.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338525.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338524.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338523.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338522.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338521.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338520.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338519.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338518.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338517.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338516.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338515.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338514.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338513.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338512.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338511.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338510.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338509.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338508.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338507.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338506.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338505.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338504.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338503.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338502.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338501.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338500.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338499.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338498.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338496.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338495.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338494.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338493.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338492.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338491.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338490.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338488.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338487.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338486.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338485.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338484.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338483.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338482.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338481.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338480.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338479.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338478.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338477.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338476.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338475.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338474.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338473.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338472.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338471.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338470.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338469.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338468.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338467.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338466.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338465.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338464.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338463.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338462.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338461.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338460.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338459.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338458.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338457.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338456.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338455.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338454.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338453.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338452.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338451.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338450.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338449.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338448.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338447.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338446.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338445.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338444.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338443.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338442.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338441.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338440.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338439.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338438.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338437.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338436.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338435.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338434.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338433.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338432.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338431.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338430.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338429.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338428.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338427.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338425.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338424.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338423.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338422.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338421.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338420.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338419.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338418.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338417.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338416.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338415.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338414.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338413.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338412.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338411.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338410.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338409.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338408.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338407.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338406.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338405.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338404.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338402.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338401.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338400.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338399.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338398.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338396.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338395.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338394.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338393.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338392.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338391.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338390.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338389.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338388.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338386.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338385.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338384.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338383.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338382.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338381.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338380.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338379.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338378.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338377.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338376.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338375.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338374.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338373.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338372.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338371.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338370.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338369.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338368.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338367.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338366.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338365.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338364.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338363.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338362.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338361.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338360.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338359.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338357.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338356.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338355.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338354.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338353.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338350.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338349.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338348.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338347.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338346.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338345.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338344.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338343.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338342.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338341.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338340.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338339.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338338.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338337.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338336.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338335.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338334.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338333.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338332.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338331.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338330.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338329.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338328.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338326.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338325.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338324.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338323.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338322.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338321.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338320.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338319.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338318.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338317.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338316.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338315.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338314.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338313.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338312.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338311.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338310.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338309.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338308.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338307.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338306.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338305.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338304.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338303.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338302.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338301.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338300.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338299.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338298.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338297.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338296.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338295.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338294.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338293.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338292.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338291.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338290.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338289.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338288.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338287.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338286.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338285.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338284.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338283.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338282.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338281.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338280.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338279.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338278.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338277.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338276.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338275.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338274.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338273.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338272.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338271.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338270.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338269.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338268.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338267.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338266.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338265.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338264.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338263.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338262.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338261.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338260.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338259.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338258.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338257.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338256.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338255.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338254.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338253.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338252.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338251.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338250.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338249.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338247.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338246.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338245.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338244.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338242.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338241.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338240.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338239.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338238.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338237.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338236.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338235.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338234.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338233.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338232.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338231.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338230.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338229.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338228.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338227.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338226.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338225.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338224.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338223.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338222.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338221.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338220.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338219.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338218.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338217.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338216.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338215.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338214.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338213.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338212.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338211.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338210.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338209.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338208.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338207.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338206.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338205.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338204.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338202.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338201.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338200.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338199.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338198.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338197.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338196.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338195.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338194.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338193.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338192.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338191.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338190.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338189.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338188.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338187.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338186.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338185.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338184.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338183.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338181.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338180.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338178.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338177.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338176.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338175.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338174.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338173.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338172.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338171.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338170.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338169.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338168.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338167.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338166.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338165.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338164.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338163.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338162.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338161.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338160.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338159.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338158.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338157.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338156.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338155.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338154.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338153.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338152.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338151.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338150.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338149.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338148.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338147.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338146.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338145.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338144.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338143.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338142.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338141.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338140.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338139.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338138.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338137.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338136.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338135.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338134.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338133.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338131.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338130.jpg',
                         u'https://cs.stanford.edu/people/rak248/VG_100K/2338129.jpg']

    # mini_entities_lst = []
    return bad_lst + mini_entities_lst


def get_bad_urls():
    """
    Get the bad urls which crashes training and testing
    :return: a list of urls
    """
    return ["https://cs.stanford.edu/people/rak248/VG_100K/2321818.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/2334844.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/3807.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2410658.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/2374264.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/2317411.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/617.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2395526.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2387785.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/150325.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/2367629.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2414329.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/1320.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/498239.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2417558.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2385742.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K/2338230.jpg",
            "https://cs.stanford.edu/people/rak248/VG_100K_2/2412.jpg"]


def get_mini_entities_img_ids():
    """
    This function returns the 1000 image_ids by FEI-FEI paper
    :return:
    """
    img_ids = [2339172, 2339171, 2339170, 2339169, 2339168, 2339167, 2339166, 2339165, 2339164, 2339163, 2339162,
               2339161, 2339160, 2339159, 2339158, 2339157, 2339156, 2339155, 2339154, 2339153, 2339152, 2339151,
               2339150, 2339149, 2339148, 2339147, 2339146, 2339145, 2339144, 2339143, 2339142, 2339141, 2339140,
               2339139, 2339138, 2339137, 2339136, 2339135, 2339134, 2339133, 2339132, 2339131, 2339130, 2339129,
               2339127, 2339126, 2339125, 2339124, 2339123, 2339122, 2339121, 2339120, 2339119, 2339118, 2339117,
               2339116, 2339115, 2339114, 2339113, 2339112, 2339110, 2339109, 2339108, 2339107, 2339106, 2339105,
               2339104, 2339103, 2339102, 2339101, 2339100, 2339099, 2339098, 2339097, 2339096, 2339095, 2339094,
               2339093, 2339092, 2339091, 2339090, 2339089, 2339088, 2339087, 2339086, 2339085, 2339084, 2339083,
               2339082, 2339081, 2339080, 2339079, 2339078, 2339077, 2339076, 2339075, 2339074, 2339073, 2339072,
               2339071, 2339070, 2339069, 2339068, 2339067, 2339066, 2339065, 2339064, 2339063, 2339062, 2339061,
               2339060, 2339059, 2339058, 2339057, 2339056, 2339055, 2339054, 2339053, 2339052, 2339051, 2339050,
               2339049, 2339048, 2339047, 2339046, 2339045, 2339044, 2339043, 2339042, 2339041, 2339040, 2339039,
               2339037, 2339036, 2339034, 2339033, 2339031, 2339030, 2339029, 2339028, 2339027, 2339026, 2339025,
               2339024, 2339023, 2339022, 2339021, 2339020, 2339019, 2339018, 2339017, 2339016, 2339015, 2339014,
               2339013, 2339012, 2339011, 2339010, 2339009, 2339008, 2339007, 2339006, 2339005, 2339004, 2339003,
               2339002, 2339001, 2339000, 2338999, 2338998, 2338997, 2338996, 2338995, 2338994, 2338993, 2338992,
               2338991, 2338990, 2338989, 2338988, 2338987, 2338986, 2338985, 2338984, 2338983, 2338982, 2338981,
               2338980, 2338979, 2338978, 2338977, 2338976, 2338975, 2338974, 2338973, 2338972, 2338971, 2338970,
               2338969, 2338968, 2338967, 2338966, 2338965, 2338964, 2338963, 2338962, 2338961, 2338960, 2338959,
               2338958, 2338957, 2338956, 2338955, 2338954, 2338953, 2338952, 2338951, 2338950, 2338949, 2338948,
               2338947, 2338946, 2338945, 2338944, 2338943, 2338942, 2338941, 2338940, 2338939, 2338938, 2338937,
               2338936, 2338935, 2338934, 2338933, 2338932, 2338931, 2338930, 2338929, 2338928, 2338927, 2338925,
               2338924, 2338923, 2338922, 2338921, 2338920, 2338919, 2338918, 2338917, 2338916, 2338915, 2338914,
               2338913, 2338912, 2338911, 2338910, 2338909, 2338908, 2338907, 2338906, 2338905, 2338904, 2338903,
               2338902, 2338901, 2338900, 2338899, 2338898, 2338897, 2338896, 2338895, 2338894, 2338893, 2338891,
               2338890, 2338889, 2338888, 2338887, 2338886, 2338885, 2338882, 2338881, 2338880, 2338879, 2338878,
               2338877, 2338876, 2338875, 2338874, 2338873, 2338872, 2338871, 2338870, 2338869, 2338868, 2338867,
               2338866, 2338865, 2338864, 2338863, 2338862, 2338861, 2338860, 2338859, 2338858, 2338857, 2338856,
               2338855, 2338854, 2338853, 2338852, 2338851, 2338850, 2338849, 2338848, 2338847, 2338846, 2338845,
               2338844, 2338843, 2338842, 2338841, 2338840, 2338839, 2338838, 2338837, 2338836, 2338835, 2338834,
               2338833, 2338831, 2338830, 2338829, 2338828, 2338827, 2338826, 2338825, 2338824, 2338822, 2338821,
               2338820, 2338819, 2338817, 2338816, 2338815, 2338814, 2338812, 2338811, 2338810, 2338809, 2338808,
               2338807, 2338806, 2338805, 2338804, 2338803, 2338802, 2338801, 2338799, 2338798, 2338797, 2338796,
               2338795, 2338794, 2338793, 2338792, 2338790, 2338789, 2338788, 2338787, 2338786, 2338785, 2338784,
               2338783, 2338782, 2338781, 2338780, 2338779, 2338778, 2338777, 2338776, 2338775, 2338774, 2338773,
               2338772, 2338771, 2338770, 2338768, 2338767, 2338766, 2338765, 2338763, 2338762, 2338761, 2338760,
               2338759, 2338757, 2338756, 2338755, 2338754, 2338753, 2338752, 2338751, 2338750, 2338749, 2338748,
               2338746, 2338745, 2338744, 2338743, 2338742, 2338740, 2338739, 2338738, 2338737, 2338736, 2338735,
               2338734, 2338733, 2338732, 2338731, 2338730, 2338729, 2338728, 2338727, 2338726, 2338725, 2338724,
               2338723, 2338722, 2338721, 2338720, 2338719, 2338718, 2338717, 2338716, 2338715, 2338714, 2338713,
               2338712, 2338711, 2338710, 2338709, 2338708, 2338707, 2338706, 2338705, 2338704, 2338703, 2338702,
               2338701, 2338700, 2338699, 2338698, 2338697, 2338696, 2338695, 2338694, 2338693, 2338692, 2338691,
               2338690, 2338689, 2338688, 2338687, 2338686, 2338685, 2338684, 2338683, 2338682, 2338681, 2338680,
               2338679, 2338678, 2338677, 2338676, 2338675, 2338674, 2338673, 2338672, 2338671, 2338670, 2338669,
               2338668, 2338666, 2338665, 2338664, 2338663, 2338662, 2338661, 2338660, 2338659, 2338658, 2338657,
               2338656, 2338655, 2338653, 2338652, 2338651, 2338650, 2338649, 2338648, 2338647, 2338646, 2338645,
               2338644, 2338643, 2338642, 2338641, 2338640, 2338639, 2338638, 2338637, 2338636, 2338634, 2338633,
               2338632, 2338631, 2338630, 2338629, 2338628, 2338627, 2338626, 2338625, 2338624, 2338621, 2338620,
               2338619, 2338618, 2338617, 2338616, 2338615, 2338614, 2338613, 2338612, 2338611, 2338610, 2338609,
               2338608, 2338607, 2338606, 2338605, 2338603, 2338602, 2338601, 2338600, 2338599, 2338598, 2338597,
               2338596, 2338595, 2338594, 2338593, 2338592, 2338591, 2338590, 2338589, 2338588, 2338587, 2338586,
               2338585, 2338584, 2338583, 2338582, 2338581, 2338580, 2338578, 2338577, 2338575, 2338574, 2338573,
               2338572, 2338571, 2338570, 2338569, 2338568, 2338567, 2338566, 2338565, 2338564, 2338563, 2338562,
               2338561, 2338560, 2338559, 2338558, 2338557, 2338556, 2338555, 2338554, 2338553, 2338552, 2338551,
               2338550, 2338549, 2338548, 2338547, 2338546, 2338545, 2338544, 2338543, 2338542, 2338541, 2338540,
               2338539, 2338538, 2338537, 2338536, 2338535, 2338534, 2338533, 2338532, 2338531, 2338530, 2338529,
               2338528, 2338527, 2338526, 2338525, 2338524, 2338523, 2338522, 2338521, 2338520, 2338519, 2338518,
               2338517, 2338516, 2338515, 2338514, 2338513, 2338512, 2338511, 2338510, 2338509, 2338508, 2338507,
               2338506, 2338505, 2338504, 2338503, 2338502, 2338501, 2338500, 2338499, 2338498, 2338496, 2338495,
               2338494, 2338493, 2338492, 2338491, 2338490, 2338488, 2338487, 2338486, 2338485, 2338484, 2338483,
               2338482, 2338481, 2338480, 2338479, 2338478, 2338477, 2338476, 2338475, 2338474, 2338473, 2338472,
               2338471, 2338470, 2338469, 2338468, 2338467, 2338466, 2338465, 2338464, 2338463, 2338462, 2338461,
               2338460, 2338459, 2338458, 2338457, 2338456, 2338455, 2338454, 2338453, 2338452, 2338451, 2338450,
               2338449, 2338448, 2338447, 2338446, 2338445, 2338444, 2338443, 2338442, 2338441, 2338440, 2338439,
               2338438, 2338437, 2338436, 2338435, 2338434, 2338433, 2338432, 2338431, 2338430, 2338429, 2338428,
               2338427, 2338425, 2338424, 2338423, 2338422, 2338421, 2338420, 2338419, 2338418, 2338417, 2338416,
               2338415, 2338414, 2338413, 2338412, 2338411, 2338410, 2338409, 2338408, 2338407, 2338406, 2338405,
               2338404, 2338402, 2338401, 2338400, 2338399, 2338398, 2338396, 2338395, 2338394, 2338393, 2338392,
               2338391, 2338390, 2338389, 2338388, 2338386, 2338385, 2338384, 2338383, 2338382, 2338381, 2338380,
               2338379, 2338378, 2338377, 2338376, 2338375, 2338374, 2338373, 2338372, 2338371, 2338370, 2338369,
               2338368, 2338367, 2338366, 2338365, 2338364, 2338363, 2338362, 2338361, 2338360, 2338359, 2338357,
               2338356, 2338355, 2338354, 2338353, 2338350, 2338349, 2338348, 2338347, 2338346, 2338345, 2338344,
               2338343, 2338342, 2338341, 2338340, 2338339, 2338338, 2338337, 2338336, 2338335, 2338334, 2338333,
               2338332, 2338331, 2338330, 2338329, 2338328, 2338326, 2338325, 2338324, 2338323, 2338322, 2338321,
               2338320, 2338319, 2338318, 2338317, 2338316, 2338315, 2338314, 2338313, 2338312, 2338311, 2338310,
               2338309, 2338308, 2338307, 2338306, 2338305, 2338304, 2338303, 2338302, 2338301, 2338300, 2338299,
               2338298, 2338297, 2338296, 2338295, 2338294, 2338293, 2338292, 2338291, 2338290, 2338289, 2338288,
               2338287, 2338286, 2338285, 2338284, 2338283, 2338282, 2338281, 2338280, 2338279, 2338278, 2338277,
               2338276, 2338275, 2338274, 2338273, 2338272, 2338271, 2338270, 2338269, 2338268, 2338267, 2338266,
               2338265, 2338264, 2338263, 2338262, 2338261, 2338260, 2338259, 2338258, 2338257, 2338256, 2338255,
               2338254, 2338253, 2338252, 2338251, 2338250, 2338249, 2338247, 2338246, 2338245, 2338244, 2338242,
               2338241, 2338240, 2338239, 2338238, 2338237, 2338236, 2338235, 2338234, 2338233, 2338232, 2338231,
               2338230, 2338229, 2338228, 2338227, 2338226, 2338225, 2338224, 2338223, 2338222, 2338221, 2338220,
               2338219, 2338218, 2338217, 2338216, 2338215, 2338214, 2338213, 2338212, 2338211, 2338210, 2338209,
               2338208, 2338207, 2338206, 2338205, 2338204, 2338202, 2338201, 2338200, 2338199, 2338198, 2338197,
               2338196, 2338195, 2338194, 2338193, 2338192, 2338191, 2338190, 2338189, 2338188, 2338187, 2338186,
               2338185, 2338184, 2338183, 2338181, 2338180, 2338178, 2338177, 2338176, 2338175, 2338174, 2338173,
               2338172, 2338171, 2338170, 2338169, 2338168, 2338167, 2338166, 2338165, 2338164, 2338163, 2338162,
               2338161, 2338160, 2338159, 2338158, 2338157, 2338156, 2338155, 2338154, 2338153, 2338152, 2338151,
               2338150, 2338149, 2338148, 2338147, 2338146, 2338145, 2338144, 2338143, 2338142, 2338141, 2338140,
               2338139, 2338138, 2338137, 2338136, 2338135, 2338134, 2338133, 2338131, 2338130, 2338129]

    return img_ids


def get_dev_entities_img_ids():
    """
    Returns the list of dev entities img ids
    :return:
    """
    return [2349648, 2349647, 2349646, 2349645, 2349644, 2349643, 2349642, 2349641, 2349640, 2349639, 2349638, 2349637,
            2349636, 2349635, 2349634, 2349633, 2349632, 2349631, 2349630, 2349629, 2349628, 2349627, 2349626, 2349624,
            2349623, 2349622, 2349621, 2349620, 2349618, 2349617, 2349616, 2349615, 2349614, 2349613, 2349612, 2349611,
            2349610, 2349609, 2349608, 2349607, 2349606, 2349605, 2349604, 2349603, 2349602, 2349601, 2349600, 2349599,
            2349598, 2349597, 2349596, 2349595, 2349594, 2349593, 2349592, 2349591, 2349590, 2349589, 2349588, 2349587,
            2349586, 2349585, 2349584, 2349583, 2349582, 2349581, 2349580, 2349579, 2349578, 2349577, 2349576, 2349575,
            2349574, 2349573, 2349572, 2349571, 2349570, 2349569, 2349568, 2349567, 2349566, 2349564, 2349563, 2349561,
            2349560, 2349559, 2349558, 2349557, 2349556, 2349555, 2349554, 2349553, 2349552, 2349551, 2349550, 2349549,
            2349548, 2349547, 2349546, 2349545, 2349544, 2349543, 2349542, 2349541, 2349540, 2349539, 2349537, 2349536,
            2349534, 2349533, 2349532, 2349531, 2349530, 2349529, 2349528, 2349527, 2349526, 2349525, 2349524, 2349523,
            2349522, 2349521, 2349520, 2349519, 2349518, 2349517, 2349516, 2349515, 2349514, 2349513, 2349512, 2349511,
            2349510, 2349509, 2349508, 2349507, 2349506, 2349505, 2349504, 2349503, 2349502, 2349501, 2349500, 2349499,
            2349498, 2349497, 2349496, 2349495, 2349494, 2349493, 2349492, 2349491, 2349490, 2349489, 2349488, 2349487,
            2349486, 2349485, 2349484, 2349483, 2349482, 2349481, 2349480, 2349479, 2349478, 2349477, 2349476, 2349475,
            2349474, 2349473, 2349472, 2349471, 2349470, 2349469, 2349468, 2349466, 2349465, 2349464, 2349463, 2349462,
            2349461, 2349460, 2349459, 2349458, 2349457, 2349456, 2349455, 2349454, 2349453, 2349452, 2349451, 2349450,
            2349449, 2349448, 2349447, 2349446, 2349445, 2349444, 2349443, 2349442, 2349441, 2349440, 2349439, 2349438,
            2349437, 2349436, 2349435, 2349434, 2349433, 2349432, 2349431, 2349429, 2349428, 2349427, 2349426, 2349425,
            2349424, 2349423, 2349422, 2349421, 2349420, 2349419, 2349418, 2349417, 2349416, 2349415, 2349414, 2349413,
            2349412, 2349411, 2349410, 2349409, 2349408, 2349407, 2349406, 2349405, 2349404, 2349403, 2349402, 2349401,
            2349400, 2349398, 2349397, 2349396, 2349395, 2349394, 2349393, 2349392, 2349391, 2349390, 2349388, 2349387,
            2349386, 2349385, 2349384, 2349383, 2349382, 2349381, 2349380, 2349379, 2349378, 2349377, 2349376, 2349375,
            2349374, 2349373, 2349372, 2349371, 2349370, 2349369, 2349368, 2349367, 2349366, 2349365, 2349364, 2349363,
            2349362, 2349361, 2349360, 2349359, 2349358, 2349357, 2349356, 2349355, 2349354, 2349353, 2349352, 2349351,
            2349350, 2349349, 2349348, 2349347, 2349346, 2349345, 2349344, 2349343, 2349342, 2349341, 2349340, 2349339,
            2349338, 2349337, 2349336, 2349335, 2349334, 2349332, 2349331, 2349330, 2349329, 2349328, 2349327, 2349326,
            2349325, 2349324, 2349323, 2349322, 2349321, 2349320, 2349319, 2349318, 2349317, 2349316, 2349315, 2349314,
            2349313, 2349312, 2349311, 2349310, 2349309, 2349308, 2349307, 2349306, 2349305, 2349304, 2349303, 2349302,
            2349301, 2349300, 2349299, 2349298, 2349297, 2349296, 2349295, 2349294, 2349293, 2349292, 2349291, 2349290,
            2349289, 2349288, 2349287, 2349286, 2349285, 2349284, 2349283, 2349282, 2349281, 2349280, 2349279, 2349278,
            2349277, 2349276, 2349275, 2349274, 2349273, 2349272, 2349271, 2349270, 2349269, 2349268, 2349267, 2349266,
            2349264, 2349263, 2349262, 2349261, 2349260, 2349259, 2349258, 2349257, 2349256, 2349255, 2349254, 2349253,
            2349252, 2349251, 2349250, 2349249, 2349248, 2349247, 2349246, 2349245, 2349244, 2349243, 2349242, 2349241,
            2349240, 2349239, 2349237, 2349236, 2349235, 2349234, 2349233, 2349232, 2349231, 2349230, 2349229, 2349228,
            2349227, 2349226, 2349225, 2349224, 2349223, 2349222, 2349221, 2349220, 2349219, 2349218, 2349217, 2349216,
            2349215, 2349214, 2349213, 2349211, 2349210, 2349209, 2349208, 2349207, 2349206, 2349205, 2349204, 2349203,
            2349202, 2349201, 2349199, 2349198, 2349197, 2349196, 2349195, 2349194, 2349193, 2349192, 2349191, 2349190,
            2349189, 2349188, 2349187, 2349186, 2349185, 2349184, 2349183, 2349182, 2349180, 2349179, 2349178, 2349177,
            2349176, 2349175, 2349174, 2349173, 2349172, 2349171, 2349170, 2349169, 2349168, 2349167, 2349166, 2349165,
            2349164, 2349163, 2349162, 2349161, 2349160, 2349159, 2349158, 2349157, 2349156, 2349155, 2349154, 2349153,
            2349152, 2349151, 2349150, 2349149, 2349148, 2349147, 2349146, 2349145, 2349143, 2349142, 2349141, 2349140,
            2349139, 2349138, 2349137, 2349136, 2349135, 2349134, 2349133, 2349132, 2349131, 2349130, 2349129, 2349128,
            2349127, 2349126, 2349125, 2349124, 2349123, 2349122, 2349121, 2349120, 2349118, 2349117, 2349116, 2349115,
            2349114, 2349113, 2349112, 2349111, 2349110, 2349109, 2349108, 2349107, 2349106, 2349105, 2349104, 2349103,
            2349102, 2349101, 2349100, 2349099, 2349098, 2349097, 2349096, 2349095, 2349094, 2349093, 2349092, 2349091,
            2349090, 2349089, 2349088, 2349087, 2349086, 2349085, 2349084, 2349083, 2349082, 2349081, 2349080, 2349079,
            2349078, 2349077, 2349076, 2349075, 2349074, 2349073, 2349072, 2349071, 2349070, 2349069, 2349068, 2349067,
            2349066, 2349065, 2349064, 2349063, 2349062, 2349061, 2349059, 2349058, 2349057, 2349054, 2349053, 2349052,
            2349051, 2349050, 2349049, 2349048, 2349047, 2349046, 2349045, 2349043, 2349042, 2349041, 2349040, 2349039,
            2349038, 2349037, 2349036, 2349035, 2349034, 2349033, 2349032, 2349031, 2349030, 2349029, 2349028, 2349027,
            2349026, 2349025, 2349024, 2349023, 2349022, 2349021, 2349020, 2349019, 2349018, 2349017, 2349016, 2349015,
            2349014, 2349013, 2349012, 2349011, 2349010, 2349009, 2349008, 2349007, 2349006, 2349005, 2349004, 2349003,
            2349002, 2349001, 2349000, 2348999, 2348998, 2348997, 2348996, 2348995, 2348994, 2348993, 2348992, 2348991,
            2348990, 2348989, 2348988, 2348987, 2348986, 2348985, 2348984, 2348983, 2348982, 2348981, 2348980, 2348979,
            2348978, 2348977, 2348976, 2348975, 2348974, 2348973, 2348972, 2348971, 2348969, 2348968, 2348967, 2348966,
            2348965, 2348964, 2348963, 2348962, 2348961, 2348960, 2348959, 2348958, 2348957, 2348956, 2348955, 2348954,
            2348953, 2348952, 2348951, 2348950, 2348949, 2348948, 2348947, 2348946, 2348945, 2348944, 2348943, 2348941,
            2348940, 2348939, 2348938, 2348937, 2348936, 2348935, 2348934, 2348933, 2348932, 2348931, 2348930, 2348929,
            2348928, 2348927, 2348926, 2348925, 2348924, 2348923, 2348922, 2348921, 2348920, 2348919, 2348918, 2348917,
            2348916, 2348915, 2348914, 2348913, 2348912, 2348911, 2348910, 2348909, 2348908, 2348907, 2348906, 2348905,
            2348904, 2348903, 2348902, 2348901, 2348900, 2348899, 2348898, 2348897, 2348896, 2348895, 2348894, 2348893,
            2348892, 2348891, 2348890, 2348889, 2348888, 2348887, 2348886, 2348885, 2348884, 2348883, 2348882, 2348881,
            2348880, 2348879, 2348878, 2348877, 2348876, 2348875, 2348873, 2348872, 2348871, 2348870, 2348869, 2348868,
            2348867, 2348866, 2348865, 2348864, 2348863, 2348862, 2348861, 2348860, 2348859, 2348858, 2348857, 2348856,
            2348855, 2348854, 2348853, 2348852, 2348851, 2348850, 2348849, 2348848, 2348847, 2348846, 2348845, 2348844,
            2348843, 2348842, 2348841, 2348840, 2348839, 2348838, 2348837, 2348836, 2348835, 2348833, 2348832, 2348831,
            2348830, 2348829, 2348828, 2348827, 2348826, 2348825, 2348824, 2348823, 2348822, 2348821, 2348820, 2348819,
            2348818, 2348817, 2348816, 2348815, 2348814, 2348813, 2348812, 2348811, 2348810, 2348809, 2348808, 2348807,
            2348806, 2348805, 2348804, 2348803, 2348802, 2348800, 2348799, 2348798, 2348797, 2348796, 2348795, 2348792,
            2348791, 2348790, 2348788, 2348787, 2348786, 2348785, 2348784, 2348783, 2348782, 2348781, 2348780, 2348779,
            2348778, 2348777, 2348776, 2348775, 2348774, 2348773, 2348772, 2348771, 2348770, 2348769, 2348768, 2348767,
            2348766, 2348765, 2348764, 2348763, 2348762, 2348761, 2348760, 2348759, 2348758, 2348757, 2348756, 2348755,
            2348754, 2348753, 2348752, 2348751, 2348750, 2348749, 2348748, 2348747, 2348746, 2348745, 2348744, 2348742,
            2348741, 2348740, 2348739, 2348738, 2348737, 2348736, 2348735, 2348734, 2348733, 2348732, 2348731, 2348730,
            2348729, 2348728, 2348727, 2348726, 2348725, 2348724, 2348723, 2348722, 2348721, 2348720, 2348719, 2348718,
            2348717, 2348716, 2348715, 2348714, 2348712, 2348711, 2348710, 2348709, 2348708, 2348707, 2348706, 2348705,
            2348704, 2348703, 2348702, 2348701, 2348700, 2348699, 2348698, 2348697, 2348696, 2348695, 2348694, 2348692,
            2348691, 2348690, 2348689, 2348688, 2348687, 2348686, 2348685, 2348684, 2348683, 2348682, 2348681, 2348680,
            2348679, 2348678, 2348677, 2348675, 2348674, 2348673, 2348672, 2348671, 2348670, 2348669, 2348667, 2348666,
            2348665, 2348664, 2348663, 2348662, 2348661, 2348660, 2348659, 2348658, 2348657, 2348655, 2348654, 2348653,
            2348651, 2348650, 2348649, 2348648, 2348647, 2348646, 2348645, 2348644, 2348643, 2348642, 2348641, 2348640,
            2348639, 2348638, 2348637, 2348636, 2348634, 2348633, 2348632, 2348631, 2348630, 2348629, 2348628, 2348627,
            2348626, 2348625, 2348624, 2348623, 2348622, 2348621, 2348620, 2348619, 2348618, 2348617, 2348616, 2348615,
            2348614, 2348613, 2348612, 2348611, 2348610, 2348609, 2348608, 2348607, 2348606, 2348605, 2348604, 2348603,
            2348602, 2348601, 2348600, 2348598, 2348597, 2348596, 2348595, 2348594, 2348593, 2348592, 2348591, 2348589,
            2348588, 2348587, 2348586, 2348584, 2348583, 2348582, 2348581, 2348580, 2348579, 2348578, 2348577, 2348576,
            2348575, 2348574, 2348573, 2348572, 2348571, 2348570, 2348569, 2348568, 2348567, 2348566, 2348565, 2348564,
            2348563, 2348562, 2348561, 2348560, 2348559, 2348558, 2348557, 2348556, 2348555, 2348554, 2348553, 2348552,
            2348551, 2348550, 2348549, 2348548, 2348547, 2348546, 2348545, 2348544, 2348543, 2348542, 2348541, 2348540,
            2348539, 2348538, 2348537, 2348536, 2348535, 2348534, 2348533, 2348532, 2348531, 2348530, 2348529, 2348528,
            2348527, 2348526, 2348525, 2348524, 2348523, 2348522, 2348521, 2348520, 2348519, 2348518, 2348517, 2348516,
            2348515, 2348514, 2348513, 2348512, 2348511, 2348510, 2348509, 2348508, 2348507, 2348506, 2348505, 2348504,
            2348503, 2348502, 2348501, 2348500, 2348499, 2348498, 2348497, 2348496, 2348495, 2348494, 2348493, 2348492,
            2348491, 2348490, 2348489, 2348488, 2348487, 2348486, 2348485, 2348484, 2348483, 2348482, 2348481, 2348480,
            2348478, 2348477, 2348476, 2348475, 2348474, 2348473, 2348471, 2348470, 2348469, 2348468, 2348467, 2348466,
            2348465, 2348463, 2348462, 2348461, 2348460, 2348459, 2348458, 2348457, 2348456, 2348455, 2348454, 2348453,
            2348452, 2348451, 2348450, 2348449, 2348448, 2348447, 2348446, 2348444, 2348443, 2348442, 2348441, 2348440,
            2348439, 2348438, 2348437, 2348436, 2348435, 2348434, 2348433, 2348432, 2348431, 2348430, 2348429, 2348428,
            2348427, 2348426, 2348425, 2348424, 2348423, 2348422, 2348421, 2348420, 2348419, 2348418, 2348417, 2348416,
            2348415, 2348414, 2348413, 2348412, 2348411, 2348410, 2348409, 2348408, 2348407, 2348406, 2348404, 2348403,
            2348402, 2348401, 2348400, 2348399, 2348398, 2348397, 2348396, 2348395, 2348394, 2348393, 2348392, 2348391,
            2348390, 2348389, 2348387, 2348386, 2348385, 2348384, 2348383, 2348382, 2348381, 2348380, 2348379, 2348378,
            2348377, 2348375, 2348374, 2348373, 2348372, 2348371, 2348370, 2348369, 2348368, 2348367, 2348366, 2348365,
            2348364, 2348363, 2348362, 2348361, 2348360, 2348359, 2348358, 2348357, 2348356, 2348355, 2348354, 2348353,
            2348352, 2348351, 2348350, 2348349, 2348348, 2348347, 2348346, 2348345, 2348344, 2348343, 2348342, 2348341,
            2348340, 2348339, 2348338, 2348337, 2348335, 2348334, 2348333, 2348332, 2348331, 2348330, 2348329, 2348328,
            2348327, 2348326, 2348325, 2348324, 2348323, 2348321, 2348320, 2348319, 2348318, 2348317, 2348316, 2348315,
            2348314, 2348313, 2348312, 2348311, 2348310, 2348309, 2348308, 2348307, 2348306, 2348305, 2348304, 2348303,
            2348302, 2348301, 2348300, 2348299, 2348298, 2348297, 2348296, 2348295, 2348294, 2348293, 2348292, 2348291,
            2348290, 2348289, 2348288, 2348287, 2348285, 2348284, 2348283, 2348282, 2348281, 2348280, 2348279, 2348278,
            2348277, 2348276, 2348275, 2348274, 2348273, 2348272, 2348271, 2348270, 2348269, 2348268, 2348267, 2348266,
            2348265, 2348264, 2348263, 2348262, 2348261, 2348260, 2348259, 2348258, 2348257, 2348256, 2348255, 2348254,
            2348253, 2348252, 2348251, 2348250, 2348249, 2348247, 2348246, 2348245, 2348244, 2348243, 2348241, 2348240,
            2348239, 2348237, 2348236, 2348235, 2348234, 2348233, 2348232, 2348231, 2348230, 2348229, 2348228, 2348227,
            2348226, 2348225, 2348223, 2348222, 2348221, 2348220, 2348219, 2348218, 2348217, 2348215, 2348214, 2348213,
            2348212, 2348211, 2348210, 2348209, 2348208, 2348207, 2348206, 2348205, 2348203, 2348202, 2348201, 2348200,
            2348199, 2348198, 2348197, 2348196, 2348195, 2348194, 2348193, 2348192, 2348191, 2348190, 2348189, 2348188,
            2348187, 2348185, 2348184, 2348183, 2348182, 2348181, 2348180, 2348179, 2348178, 2348177, 2348176, 2348175,
            2348174, 2348173, 2348172, 2348171, 2348170, 2348169, 2348168, 2348167, 2348166, 2348165, 2348164, 2348163,
            2348162, 2348160, 2348159, 2348158, 2348157, 2348156, 2348155, 2348154, 2348153, 2348152, 2348151, 2348150,
            2348149, 2348147, 2348146, 2348144, 2348143, 2348142, 2348141, 2348140, 2348139, 2348138, 2348137, 2348136,
            2348135, 2348134, 2348133, 2348132, 2348131, 2348130, 2348129, 2348128, 2348127, 2348126, 2348125, 2348124,
            2348123, 2348122, 2348121, 2348120, 2348119, 2348118, 2348117, 2348116, 2348115, 2348114, 2348113, 2348112,
            2348111, 2348110, 2348109, 2348108, 2348107, 2348106, 2348105, 2348104, 2348102, 2348101, 2348100, 2348099,
            2348098, 2348097, 2348095, 2348094, 2348093, 2348092, 2348091, 2348090, 2348089, 2348088, 2348087, 2348086,
            2348085, 2348084, 2348083, 2348082, 2348080, 2348079, 2348078, 2348077, 2348076, 2348075, 2348074, 2348073,
            2348072, 2348071, 2348070, 2348069, 2348068, 2348067, 2348066, 2348065, 2348064, 2348063, 2348062, 2348061,
            2348059, 2348058, 2348057, 2348056, 2348055, 2348054, 2348053, 2348052, 2348051, 2348050, 2348049, 2348048,
            2348047, 2348046, 2348045, 2348044, 2348043, 2348042, 2348041, 2348040, 2348039, 2348038, 2348037, 2348036,
            2348035, 2348034, 2348033, 2348032, 2348031, 2348030, 2348028, 2348027, 2348026, 2348025, 2348024, 2348023,
            2348022, 2348021, 2348020, 2348019, 2348017, 2348016, 2348015, 2348014, 2348013, 2348012, 2348010, 2348008,
            2348007, 2348005, 2348004, 2348003, 2348002, 2348001, 2348000, 2347999, 2347998, 2347997, 2347996, 2347995,
            2347994, 2347993, 2347992, 2347990, 2347989, 2347988, 2347987, 2347986, 2347985, 2347984, 2347983, 2347982,
            2347981, 2347980, 2347979, 2347978, 2347977, 2347976, 2347975, 2347974, 2347973, 2347972, 2347971, 2347969,
            2347968, 2347967, 2347966, 2347965, 2347964, 2347963, 2347962, 2347961, 2347960, 2347959, 2347958, 2347957,
            2347956, 2347955, 2347954, 2347953, 2347952, 2347951, 2347950, 2347949, 2347948, 2347946, 2347945, 2347944,
            2347943, 2347942, 2347941, 2347940, 2347939, 2347938, 2347937, 2347936, 2347935, 2347934, 2347933, 2347932,
            2347930, 2347929, 2347928, 2347927, 2347926, 2347925, 2347924, 2347923, 2347922, 2347921, 2347920, 2347919,
            2347918, 2347917, 2347916, 2347915, 2347914, 2347913, 2347912, 2347911, 2347910, 2347909, 2347908, 2347907,
            2347906, 2347905, 2347904, 2347902, 2347901, 2347900, 2347899, 2347898, 2347897, 2347896, 2347895, 2347894,
            2347893, 2347892, 2347891, 2347890, 2347889, 2347888, 2347887, 2347886, 2347885, 2347884, 2347883, 2347882,
            2347881, 2347880, 2347879, 2347878, 2347877, 2347876, 2347875, 2347874, 2347873, 2347871, 2347870, 2347869,
            2347868, 2347867, 2347866, 2347865, 2347863, 2347862, 2347861, 2347860, 2347858, 2347857, 2347856, 2347855,
            2347854, 2347853, 2347852, 2347851, 2347849, 2347848, 2347847, 2347846, 2347845, 2347844, 2347843, 2347842,
            2347841, 2347840, 2347839, 2347838, 2347837, 2347836, 2347835, 2347834, 2347833, 2347832, 2347831, 2347830,
            2347829, 2347828, 2347827, 2347826, 2347825, 2347824, 2347823, 2347822, 2347821, 2347820, 2347819, 2347818,
            2347817, 2347816, 2347815, 2347814, 2347813, 2347812, 2347811, 2347810, 2347809, 2347808, 2347807, 2347806,
            2347805, 2347804, 2347803, 2347802, 2347801, 2347800, 2347799, 2347798, 2347797, 2347796, 2347795, 2347794,
            2347793, 2347792, 2347791, 2347790, 2347789, 2347788, 2347787, 2347786, 2347785, 2347784, 2347783, 2347782,
            2347781, 2347780, 2347779, 2347778, 2347777, 2347775, 2347774, 2347773, 2347772, 2347771, 2347770, 2347769,
            2347768, 2347767, 2347766, 2347765, 2347763, 2347761, 2347760, 2347759, 2347758, 2347757, 2347756, 2347755,
            2347754, 2347753, 2347752, 2347750, 2347749, 2347748, 2347747, 2347746, 2347745, 2347744, 2347743, 2347742,
            2347741, 2347740, 2347739, 2347738, 2347737, 2347736, 2347735, 2347734, 2347733, 2347731, 2347730, 2347729,
            2347728, 2347727, 2347726, 2347725, 2347724, 2347723, 2347722, 2347721, 2347720, 2347719, 2347718, 2347717,
            2347716, 2347715, 2347714, 2347713, 2347712, 2347711, 2347710, 2347709, 2347708, 2347707, 2347706, 2347705,
            2347704, 2347703, 2347702, 2347701, 2347700, 2347699, 2347698, 2347697, 2347696, 2347695, 2347694, 2347693,
            2347692, 2347691, 2347690, 2347689, 2347688, 2347687, 2347686, 2347685, 2347684, 2347683, 2347682, 2347681,
            2347680, 2347679, 2347678, 2347677, 2347676, 2347675, 2347674, 2347673, 2347672, 2347671, 2347670, 2347669,
            2347668, 2347667, 2347666, 2347665, 2347664, 2347663, 2347662, 2347661, 2347660, 2347659, 2347658, 2347657,
            2347656, 2347655, 2347654, 2347653, 2347652, 2347651, 2347650, 2347649, 2347648, 2347647, 2347646, 2347645,
            2347643, 2347642, 2347641, 2347640, 2347639, 2347638, 2347637, 2347636, 2347635, 2347634, 2347633, 2347632,
            2347631, 2347630, 2347629, 2347628, 2347627, 2347626, 2347625, 2347623, 2347622, 2347621, 2347620, 2347619,
            2347618, 2347617, 2347616, 2347615, 2347614, 2347613, 2347612, 2347611, 2347610, 2347609, 2347608, 2347607,
            2347606, 2347605, 2347604, 2347602, 2347601, 2347600, 2347599, 2347598, 2347597, 2347596, 2347595, 2347594,
            2347593, 2347592, 2347591, 2347590, 2347589, 2347588, 2347587, 2347586, 2347585, 2347584, 2347583, 2347582,
            2347581, 2347580, 2347578, 2347577, 2347576, 2347575, 2347574, 2347572, 2347571, 2347570, 2347569, 2347568,
            2347567, 2347566, 2347565, 2347564, 2347563, 2347562, 2347561, 2347560, 2347559, 2347557, 2347556, 2347555,
            2347554, 2347553, 2347552, 2347551, 2347550, 2347549, 2347548, 2347547, 2347546, 2347545, 2347544, 2347543,
            2347541, 2347540, 2347539, 2347538, 2347537, 2347535, 2347534, 2347533, 2347532, 2347531, 2347530, 2347529,
            2347528, 2347527, 2347526, 2347525, 2347524, 2347523, 2347522, 2347521, 2347520, 2347519, 2347518, 2347517,
            2347516, 2347515, 2347514, 2347513, 2347512, 2347511, 2347510, 2347509, 2347508, 2347507, 2347506, 2347505,
            2347504, 2347503, 2347502, 2347501, 2347500, 2347499, 2347498, 2347497, 2347496, 2347495, 2347494, 2347493,
            2347492, 2347491, 2347490, 2347489, 2347488, 2347487, 2347486, 2347485, 2347484, 2347483, 2347482, 2347481,
            2347480, 2347479, 2347478, 2347477, 2347476, 2347475, 2347474, 2347473, 2347471, 2347470, 2347469, 2347468,
            2347467, 2347466, 2347465, 2347463, 2347462, 2347461, 2347460, 2347459, 2347458, 2347457, 2347456, 2347455,
            2347454, 2347453, 2347452, 2347451, 2347450, 2347449, 2347448, 2347447, 2347446, 2347445, 2347444, 2347443,
            2347442, 2347441, 2347439, 2347438, 2347437, 2347436, 2347435, 2347434, 2347433, 2347432, 2347431, 2347430,
            2347429, 2347428, 2347427, 2347426, 2347425, 2347424, 2347423, 2347422, 2347421, 2347420, 2347419, 2347418,
            2347417, 2347416, 2347415, 2347414, 2347413, 2347411, 2347410, 2347409, 2347408, 2347407, 2347406, 2347405,
            2347404, 2347403, 2347402, 2347401, 2347400, 2347399, 2347398, 2347396, 2347395, 2347394, 2347393, 2347392,
            2347391, 2347390, 2347389, 2347388, 2347387, 2347386, 2347385, 2347384, 2347383, 2347382, 2347381, 2347380,
            2347379, 2347378, 2347377, 2347375, 2347374, 2347373, 2347372, 2347371, 2347369, 2347368, 2347367, 2347366,
            2347365, 2347364, 2347363, 2347362, 2347361, 2347360, 2347359, 2347358, 2347357, 2347356, 2347355, 2347354,
            2347353, 2347352, 2347351, 2347350, 2347349, 2347348, 2347347, 2347346, 2347345, 2347344, 2347343, 2347342,
            2347341, 2347340, 2347339, 2347338, 2347337, 2347336, 2347335, 2347334, 2347333, 2347332, 2347331, 2347330,
            2347329, 2347328, 2347327, 2347326, 2347325, 2347324, 2347323, 2347322, 2347321, 2347318, 2347317, 2347316,
            2347315, 2347314, 2347313, 2347312, 2347311, 2347310, 2347309, 2347308, 2347307, 2347306, 2347305, 2347304,
            2347303, 2347302, 2347301, 2347300, 2347299, 2347298, 2347297, 2347296, 2347295, 2347294, 2347293, 2347292,
            2347291, 2347290, 2347289, 2347288, 2347287, 2347286, 2347285, 2347284, 2347283, 2347282, 2347281, 2347280,
            2347279, 2347278, 2347277, 2347276, 2347275, 2347274, 2347273, 2347272, 2347271, 2347270, 2347268, 2347267,
            2347266, 2347265, 2347264, 2347263, 2347262, 2347261, 2347260, 2347259, 2347258, 2347257, 2347256, 2347255,
            2347254, 2347253, 2347252, 2347251, 2347250, 2347249, 2347248, 2347247, 2347246, 2347245, 2347244, 2347243,
            2347242, 2347241, 2347240, 2347239, 2347238, 2347237, 2347236, 2347235, 2347234, 2347233, 2347232, 2347231,
            2347230, 2347229, 2347228, 2347226, 2347225, 2347224, 2347223, 2347222, 2347221, 2347220, 2347219, 2347218,
            2347217, 2347216, 2347215, 2347214, 2347213, 2347212, 2347211, 2347210, 2347209, 2347208, 2347206, 2347205,
            2347204, 2347203, 2347202, 2347201, 2347200, 2347199, 2347198, 2347197, 2347196, 2347195, 2347194, 2347193,
            2347192, 2347191, 2347190, 2347189, 2347188, 2347187, 2347186, 2347185, 2347184, 2347183, 2347182, 2347181,
            2347180, 2347179, 2347178, 2347177, 2347176, 2347175, 2347174, 2347173, 2347171, 2347170, 2347169, 2347168,
            2347167, 2347166, 2347165, 2347164, 2347163, 2347162, 2347160, 2347159, 2347158, 2347157, 2347156, 2347155,
            2347154, 2347153, 2347152, 2347151, 2347150, 2347149, 2347148, 2347147, 2347146, 2347145, 2347144, 2347143,
            2347142, 2347141, 2347140, 2347139, 2347138, 2347137, 2347136, 2347135, 2347134, 2347133, 2347132, 2347131,
            2347130, 2347129, 2347128, 2347127, 2347126, 2347125, 2347124, 2347123, 2347122, 2347121, 2347120, 2347119,
            2347118, 2347117, 2347116, 2347115, 2347114, 2347113, 2347112, 2347111, 2347110, 2347109, 2347108, 2347107,
            2347106, 2347105, 2347104, 2347103, 2347102, 2347101, 2347100, 2347099, 2347098, 2347097, 2347096, 2347095,
            2347094, 2347093, 2347092, 2347091, 2347090, 2347089, 2347088, 2347087, 2347086, 2347085, 2347084, 2347083,
            2347082, 2347081, 2347079, 2347078, 2347077, 2347076, 2347075, 2347074, 2347073, 2347072, 2347071, 2347070,
            2347069, 2347068, 2347066, 2347065, 2347063, 2347062, 2347061, 2347060, 2347059, 2347058, 2347057, 2347056,
            2347055, 2347054, 2347053, 2347052, 2347051, 2347050, 2347049, 2347048, 2347047, 2347046, 2347045, 2347044,
            2347043, 2347042, 2347041, 2347040, 2347039, 2347038, 2347037, 2347036, 2347035, 2347034, 2347033, 2347032,
            2347031, 2347030, 2347029, 2347028, 2347027, 2347026, 2347025, 2347024, 2347023, 2347022, 2347021, 2347020,
            2347019, 2347018, 2347017, 2347016, 2347015, 2347014, 2347012, 2347011, 2347010, 2347009, 2347008, 2347007,
            2347006, 2347005, 2347004, 2347003, 2347002, 2347001, 2347000, 2346999, 2346998, 2346997, 2346996, 2346995,
            2346994, 2346993, 2346992, 2346991, 2346990, 2346989, 2346988, 2346987, 2346986, 2346985, 2346984, 2346983,
            2346982, 2346981, 2346980, 2346979, 2346978, 2346977, 2346976, 2346975, 2346974, 2346973, 2346972, 2346971,
            2346970, 2346969, 2346968, 2346967, 2346966, 2346965, 2346964, 2346963, 2346962, 2346961, 2346960, 2346959,
            2346958, 2346957, 2346956, 2346955, 2346954, 2346952, 2346951, 2346950, 2346949, 2346948, 2346947, 2346946,
            2346945, 2346943, 2346942, 2346941, 2346940, 2346939, 2346938, 2346937, 2346935, 2346934, 2346933, 2346932,
            2346931, 2346930, 2346929, 2346928, 2346927, 2346926, 2346925, 2346923, 2346922, 2346921, 2346920, 2346918,
            2346917, 2346916, 2346915, 2346914, 2346913, 2346912, 2346911, 2346910, 2346909, 2346908, 2346907, 2346906,
            2346905, 2346904, 2346903, 2346902, 2346901, 2346900, 2346899, 2346898, 2346897, 2346896, 2346895, 2346894,
            2346893, 2346892, 2346890, 2346889, 2346888, 2346887, 2346885, 2346883, 2346882, 2346881, 2346880, 2346879,
            2346878, 2346877, 2346876, 2346875, 2346874, 2346873, 2346872, 2346871, 2346870, 2346869, 2346866, 2346865,
            2346864, 2346863, 2346862, 2346861, 2346860, 2346859, 2346858, 2346857, 2346856, 2346855, 2346854, 2346853,
            2346852, 2346851, 2346850, 2346849, 2346848, 2346847, 2346846, 2346845, 2346844, 2346843, 2346842, 2346841,
            2346840, 2346839, 2346838, 2346837, 2346836, 2346835, 2346834, 2346833, 2346832, 2346831, 2346830, 2346829,
            2346828, 2346827, 2346826, 2346825, 2346824, 2346823, 2346822, 2346821, 2346820, 2346819, 2346818, 2346817,
            2346816, 2346815, 2346814, 2346813, 2346812, 2346811, 2346810, 2346809, 2346807, 2346806, 2346805, 2346804,
            2346803, 2346802, 2346801, 2346800, 2346799, 2346798, 2346797, 2346796, 2346795, 2346794, 2346793, 2346792,
            2346791, 2346790, 2346789, 2346788, 2346787, 2346786, 2346785, 2346784, 2346783, 2346782, 2346781, 2346780,
            2346779, 2346778, 2346777, 2346776, 2346775, 2346774, 2346772, 2346771, 2346770, 2346769, 2346768, 2346767,
            2346766, 2346765, 2346764, 2346763, 2346762, 2346761, 2346760, 2346759, 2346758, 2346757, 2346756, 2346755,
            2346754, 2346753, 2346752, 2346751, 2346750, 2346749, 2346748, 2346746, 2346744, 2346743, 2346742, 2346741,
            2346740, 2346739, 2346738, 2346737, 2346736, 2346735, 2346734, 2346733, 2346732, 2346731, 2346730, 2346728,
            2346727, 2346726, 2346725, 2346723, 2346722, 2346721, 2346720, 2346719, 2346718, 2346717, 2346716, 2346715,
            2346714, 2346713, 2346712, 2346711, 2346710, 2346709, 2346708, 2346707, 2346706, 2346705, 2346704, 2346703,
            2346702, 2346701, 2346699, 2346698, 2346697, 2346696, 2346695, 2346694, 2346693, 2346692, 2346691, 2346690,
            2346689, 2346688, 2346687, 2346686, 2346685, 2346683, 2346682, 2346681, 2346679, 2346678, 2346677, 2346676,
            2346675, 2346674, 2346673, 2346672, 2346671, 2346670, 2346669, 2346668, 2346667, 2346666, 2346665, 2346664,
            2346663, 2346662, 2346661, 2346660, 2346659, 2346658, 2346657, 2346656, 2346655, 2346654, 2346653, 2346652,
            2346651, 2346650, 2346649, 2346648, 2346647, 2346646, 2346645, 2346644, 2346643, 2346642, 2346641, 2346640,
            2346639, 2346638, 2346637, 2346636, 2346635, 2346634, 2346633, 2346632, 2346631, 2346630, 2346629, 2346628,
            2346627, 2346626, 2346625, 2346624, 2346623, 2346622, 2346621, 2346620, 2346619, 2346618, 2346617, 2346616,
            2346615, 2346614, 2346613, 2346612, 2346611, 2346610, 2346609, 2346608, 2346607, 2346606, 2346605, 2346604,
            2346603, 2346602, 2346601, 2346600, 2346599, 2346598, 2346597, 2346596, 2346595, 2346593, 2346592, 2346591,
            2346590, 2346589, 2346588, 2346586, 2346585, 2346584, 2346583, 2346582, 2346581, 2346580, 2346579, 2346577,
            2346575, 2346574, 2346573, 2346572, 2346571, 2346570, 2346569, 2346568, 2346567, 2346566, 2346565, 2346564,
            2346563, 2346561, 2346560, 2346559, 2346558, 2346557, 2346556, 2346555, 2346553, 2346552, 2346551, 2346550,
            2346548, 2346547, 2346546, 2346545, 2346544, 2346543, 2346542, 2346541, 2346540, 2346539, 2346538, 2346537,
            2346536, 2346535, 2346534, 2346533, 2346532, 2346531, 2346530, 2346529, 2346528, 2346527, 2346526, 2346525,
            2346524, 2346523, 2346522, 2346521, 2346520, 2346519, 2346518, 2346517, 2346516, 2346515, 2346514, 2346513,
            2346512, 2346511, 2346510, 2346509, 2346508, 2346507, 2346506, 2346505, 2346504, 2346503, 2346502, 2346501,
            2346500, 2346499, 2346498, 2346497, 2346496, 2346495, 2346494, 2346493, 2346492, 2346491, 2346490, 2346489,
            2346488, 2346487, 2346486, 2346485, 2346484, 2346483, 2346482, 2346481, 2346480, 2346479, 2346478, 2346477,
            2346476, 2346475, 2346474, 2346473, 2346472, 2346471, 2346470, 2346469, 2346468, 2346467, 2346466, 2346465,
            2346464, 2346462, 2346461, 2346460, 2346459, 2346458, 2346457, 2346456, 2346455, 2346454, 2346453, 2346452,
            2346451, 2346450, 2346449, 2346447, 2346446, 2346445, 2346444, 2346443, 2346442, 2346441, 2346440, 2346439,
            2346438, 2346437, 2346436, 2346435, 2346434, 2346433, 2346432, 2346431, 2346430, 2346429, 2346428, 2346427,
            2346426, 2346425, 2346424, 2346423, 2346422, 2346421, 2346420, 2346419, 2346418, 2346417, 2346416, 2346415,
            2346414, 2346413, 2346412, 2346411, 2346410, 2346409, 2346408, 2346407, 2346406, 2346405, 2346404, 2346403,
            2346402, 2346401, 2346399, 2346398, 2346397, 2346396, 2346395, 2346394, 2346393, 2346392, 2346391, 2346390,
            2346389, 2346388, 2346387, 2346386, 2346385, 2346384, 2346383, 2346382, 2346381, 2346380, 2346379, 2346378,
            2346377, 2346376, 2346375, 2346374, 2346373, 2346372, 2346371, 2346370, 2346368, 2346367, 2346366, 2346365,
            2346364, 2346363, 2346362, 2346361, 2346360, 2346359, 2346358, 2346357, 2346356, 2346355, 2346354, 2346353,
            2346352, 2346351, 2346350, 2346349, 2346348, 2346347, 2346346, 2346345, 2346344, 2346343, 2346342, 2346341,
            2346340, 2346339, 2346338, 2346337, 2346336, 2346335, 2346334, 2346333, 2346332, 2346331, 2346330, 2346329,
            2346328, 2346327, 2346326, 2346324, 2346323, 2346322, 2346321, 2346320, 2346319, 2346318, 2346317, 2346316,
            2346315, 2346314, 2346313, 2346312, 2346311, 2346310, 2346308, 2346307, 2346306, 2346305, 2346304, 2346303,
            2346302, 2346301, 2346300, 2346299, 2346298, 2346297, 2346296, 2346295, 2346294, 2346292, 2346291, 2346290,
            2346289, 2346288, 2346287, 2346286, 2346285, 2346284, 2346283, 2346282, 2346281, 2346280, 2346279, 2346278,
            2346277, 2346276, 2346275, 2346274, 2346272, 2346271, 2346270, 2346269, 2346268, 2346267, 2346266, 2346265,
            2346264, 2346263, 2346262, 2346261, 2346260, 2346259, 2346258, 2346257, 2346256, 2346255, 2346254, 2346253,
            2346252, 2346251, 2346250, 2346249, 2346248, 2346247, 2346246, 2346245, 2346244, 2346243, 2346242, 2346241,
            2346240, 2346239, 2346238, 2346237, 2346236, 2346235, 2346234, 2346233, 2346232, 2346231, 2346230, 2346229,
            2346228, 2346227, 2346226, 2346225, 2346224, 2346223, 2346222, 2346221, 2346220, 2346219, 2346218, 2346217,
            2346216, 2346215, 2346214, 2346213, 2346212, 2346211, 2346210, 2346209, 2346208, 2346207, 2346206, 2346205,
            2346204, 2346202, 2346200, 2346199, 2346198, 2346197, 2346196, 2346195, 2346194, 2346193, 2346192, 2346191,
            2346190, 2346189, 2346188, 2346187, 2346186, 2346185, 2346184, 2346183, 2346182, 2346181, 2346180, 2346179,
            2346178, 2346177, 2346176, 2346175, 2346174, 2346173, 2346172, 2346171, 2346170, 2346169, 2346168, 2346167,
            2346166, 2346165, 2346164, 2346163, 2346162, 2346161, 2346160, 2346159, 2346158, 2346157, 2346156, 2346155,
            2346154, 2346153, 2346152, 2346151, 2346149, 2346148, 2346147, 2346146, 2346145, 2346144, 2346143, 2346142,
            2346141, 2346140, 2346139, 2346138, 2346137, 2346136, 2346135, 2346134, 2346133, 2346132, 2346131, 2346130,
            2346129, 2346128, 2346127, 2346126, 2346125, 2346124, 2346123, 2346122, 2346121, 2346120, 2346119, 2346117,
            2346116, 2346114, 2346113, 2346112, 2346111, 2346110, 2346109, 2346108, 2346107, 2346106, 2346105, 2346104,
            2346103, 2346102, 2346101, 2346100, 2346099, 2346098, 2346097, 2346096, 2346095, 2346094, 2346093, 2346092,
            2346091, 2346090, 2346089, 2346088, 2346087, 2346086, 2346085, 2346084, 2346083, 2346082, 2346081, 2346080,
            2346079, 2346078, 2346077, 2346076, 2346075, 2346074, 2346073, 2346072, 2346071, 2346070, 2346069, 2346068,
            2346067, 2346066, 2346064, 2346063, 2346062, 2346061, 2346060, 2346059, 2346058, 2346057, 2346056, 2346054,
            2346053, 2346052, 2346051, 2346050, 2346049, 2346048, 2346047, 2346046, 2346045, 2346044, 2346042, 2346040,
            2346039, 2346038, 2346037, 2346036, 2346035, 2346034, 2346033, 2346032, 2346031, 2346030, 2346029, 2346028,
            2346027, 2346026, 2346025, 2346024, 2346023, 2346022, 2346021, 2346020, 2346019, 2346018, 2346016, 2346015,
            2346014, 2346013, 2346012, 2346011, 2346010, 2346009, 2346008, 2346007, 2346006, 2346005, 2346004, 2346003,
            2346002, 2346001, 2346000, 2345999, 2345998, 2345997, 2345996, 2345995, 2345994, 2345993, 2345991, 2345990,
            2345989, 2345988, 2345987, 2345985, 2345984, 2345983, 2345982, 2345981, 2345980, 2345979, 2345978, 2345977,
            2345976, 2345975, 2345974, 2345973, 2345972, 2345971, 2345970, 2345969, 2345968, 2345967, 2345966, 2345965,
            2345964, 2345963, 2345962, 2345961, 2345960, 2345959, 2345958, 2345957, 2345956, 2345955, 2345954, 2345953,
            2345952, 2345951, 2345950, 2345949, 2345948, 2345947, 2345946, 2345944, 2345942, 2345941, 2345940, 2345939,
            2345938, 2345937, 2345936, 2345934, 2345933, 2345932, 2345931, 2345930, 2345929, 2345928, 2345927, 2345925,
            2345924, 2345923, 2345922, 2345921, 2345920, 2345918, 2345917, 2345916, 2345915, 2345914, 2345913, 2345912,
            2345911, 2345910, 2345907, 2345906, 2345905, 2345904, 2345903, 2345902, 2345900, 2345899, 2345898, 2345897,
            2345896, 2345895, 2345894, 2345893, 2345892, 2345891, 2345888, 2345887, 2345886, 2345885, 2345884, 2345883,
            2345882, 2345881, 2345880, 2345879, 2345878, 2345877, 2345876, 2345875, 2345874, 2345873, 2345872, 2345871,
            2345870, 2345869, 2345868, 2345867, 2345866, 2345865, 2345864, 2345863, 2345862, 2345861, 2345860, 2345859,
            2345858, 2345857, 2345856, 2345855, 2345854, 2345853, 2345852, 2345851, 2345850, 2345849, 2345848, 2345847,
            2345846, 2345845, 2345844, 2345843, 2345842, 2345841, 2345840, 2345839, 2345838, 2345837, 2345836, 2345835,
            2345834, 2345833, 2345832, 2345831, 2345830, 2345829, 2345828, 2345827, 2345826, 2345825, 2345824, 2345823,
            2345822, 2345821, 2345820, 2345819, 2345818, 2345817, 2345816, 2345815, 2345814, 2345813, 2345812, 2345811,
            2345810, 2345809, 2345808, 2345807, 2345806, 2345805, 2345804, 2345803, 2345802, 2345801, 2345800, 2345799,
            2345798, 2345797, 2345796, 2345795, 2345794, 2345793, 2345792, 2345791, 2345790, 2345789, 2345788, 2345787,
            2345786, 2345785, 2345783, 2345782, 2345781, 2345780, 2345779, 2345777, 2345776, 2345775, 2345774, 2345773,
            2345772, 2345771, 2345770, 2345769, 2345768, 2345767, 2345766, 2345765, 2345764, 2345763, 2345762, 2345761,
            2345760, 2345759, 2345758, 2345757, 2345756, 2345755, 2345754, 2345753, 2345752, 2345751, 2345750, 2345749,
            2345747, 2345746, 2345745, 2345744, 2345743, 2345742, 2345741, 2345740, 2345739, 2345738, 2345737, 2345736,
            2345735, 2345734, 2345733, 2345732, 2345731, 2345730, 2345729, 2345728, 2345727, 2345726, 2345725, 2345723,
            2345722, 2345721, 2345720, 2345719, 2345718, 2345717, 2345716, 2345715, 2345714, 2345713, 2345712, 2345711,
            2345710, 2345709, 2345708, 2345707, 2345706, 2345705, 2345704, 2345703, 2345702, 2345701, 2345700, 2345699,
            2345698, 2345697, 2345696, 2345695, 2345694, 2345693, 2345692, 2345691, 2345690, 2345689, 2345688, 2345687,
            2345686, 2345685, 2345684, 2345683, 2345682, 2345681, 2345680, 2345679, 2345678, 2345677, 2345676, 2345675,
            2345674, 2345673, 2345672, 2345671, 2345670, 2345669, 2345668, 2345667, 2345666, 2345665, 2345664, 2345663,
            2345662, 2345661, 2345660, 2345658, 2345657, 2345656, 2345655, 2345654, 2345653, 2345652, 2345651, 2345650,
            2345649, 2345648, 2345647, 2345646, 2345645, 2345644, 2345643, 2345642, 2345641, 2345640, 2345639, 2345638,
            2345637, 2345635, 2345634, 2345633, 2345632, 2345631, 2345630, 2345629, 2345628, 2345627, 2345626, 2345625,
            2345624, 2345623, 2345622, 2345621, 2345620, 2345619, 2345618, 2345617, 2345616, 2345615, 2345614, 2345613,
            2345612, 2345610, 2345609, 2345608, 2345607, 2345606, 2345605, 2345604, 2345602, 2345601, 2345600, 2345599,
            2345598, 2345597, 2345596, 2345595, 2345594, 2345593, 2345592, 2345591, 2345590, 2345589, 2345588, 2345587,
            2345586, 2345585, 2345584, 2345583, 2345582, 2345581, 2345580, 2345579, 2345578, 2345577, 2345576, 2345574,
            2345573, 2345572, 2345571, 2345570, 2345569, 2345568, 2345567, 2345566, 2345565, 2345563, 2345562, 2345561,
            2345560, 2345559, 2345558, 2345557, 2345556, 2345555, 2345554, 2345553, 2345552, 2345551, 2345550, 2345549,
            2345548, 2345547, 2345546, 2345545, 2345544, 2345543, 2345542, 2345541, 2345540, 2345539, 2345538, 2345537,
            2345536, 2345535, 2345534, 2345533, 2345532, 2345531, 2345530, 2345529, 2345528, 2345527, 2345526, 2345525,
            2345524, 2345523, 2345522, 2345521, 2345520, 2345519, 2345518, 2345516, 2345515, 2345514, 2345513, 2345512,
            2345511, 2345510, 2345509, 2345508, 2345507, 2345506, 2345505, 2345504, 2345503, 2345502, 2345501, 2345500,
            2345499, 2345498, 2345497, 2345496, 2345495, 2345494, 2345493, 2345492, 2345491, 2345490, 2345489, 2345488,
            2345487, 2345486, 2345485, 2345484, 2345483, 2345482, 2345481, 2345480, 2345479, 2345478, 2345477, 2345476,
            2345475, 2345474, 2345473, 2345471, 2345470, 2345469, 2345468, 2345467, 2345466, 2345465, 2345464, 2345462,
            2345461, 2345460, 2345459, 2345458, 2345457, 2345456, 2345455, 2345454, 2345453, 2345452, 2345451, 2345450,
            2345449, 2345448, 2345447, 2345446, 2345445, 2345444, 2345443, 2345442, 2345441, 2345440, 2345439, 2345438,
            2345437, 2345436, 2345435, 2345434, 2345433, 2345432, 2345431, 2345430, 2345429, 2345428, 2345427, 2345426,
            2345425, 2345424, 2345423, 2345422, 2345421, 2345420, 2345419, 2345418, 2345417, 2345416, 2345415, 2345413,
            2345412, 2345411, 2345410, 2345409, 2345408, 2345407, 2345406, 2345405, 2345404, 2345403, 2345402, 2345401,
            2345399, 2345398, 2345397, 2345396, 2345395, 2345394, 2345393, 2345392, 2345391, 2345390, 2345389, 2345388,
            2345387, 2345386, 2345385, 2345384, 2345383, 2345382, 2345381, 2345380, 2345379, 2345378, 2345377, 2345376,
            2345375, 2345374, 2345373, 2345372, 2345371, 2345370, 2345369, 2345368, 2345367, 2345366, 2345365, 2345364,
            2345363, 2345362, 2345361, 2345360, 2345359, 2345358, 2345357, 2345356, 2345355, 2345354, 2345353, 2345352,
            2345351, 2345350, 2345349, 2345348, 2345347, 2345346, 2345345, 2345344, 2345343, 2345342, 2345341, 2345340,
            2345339, 2345337, 2345336, 2345335, 2345334, 2345331, 2345330, 2345329, 2345328, 2345327, 2345326, 2345325,
            2345324, 2345323, 2345322, 2345321, 2345320, 2345319, 2345318, 2345317, 2345316, 2345315, 2345314, 2345313,
            2345312, 2345311, 2345310, 2345309, 2345308, 2345307, 2345306, 2345305, 2345304, 2345303, 2345302, 2345301,
            2345300, 2345299, 2345298, 2345297, 2345296, 2345295, 2345294, 2345293, 2345292, 2345291, 2345290, 2345289,
            2345288, 2345287, 2345286, 2345285, 2345284, 2345283, 2345282, 2345281, 2345280, 2345279, 2345278, 2345277,
            2345276, 2345275, 2345274, 2345273, 2345272, 2345271, 2345270, 2345269, 2345268, 2345267, 2345266, 2345265,
            2345264, 2345263, 2345262, 2345261, 2345260, 2345259, 2345258, 2345257, 2345256, 2345255, 2345254, 2345253,
            2345252, 2345251, 2345250, 2345249, 2345248, 2345247, 2345246, 2345245, 2345244, 2345243, 2345242, 2345241,
            2345240, 2345239, 2345237, 2345236, 2345234, 2345233, 2345232, 2345231, 2345230, 2345229, 2345228, 2345227,
            2345226, 2345225, 2345224, 2345223, 2345222, 2345220, 2345219, 2345218, 2345217, 2345215, 2345214, 2345213,
            2345212, 2345211, 2345210, 2345209, 2345208, 2345207, 2345205, 2345204, 2345203, 2345202, 2345201, 2345200,
            2345199, 2345198, 2345197, 2345195, 2345194, 2345193, 2345192, 2345191, 2345190, 2345189, 2345188, 2345187,
            2345186, 2345184, 2345182, 2345181, 2345180, 2345179, 2345178, 2345177, 2345176, 2345175, 2345174, 2345173,
            2345172, 2345171, 2345170, 2345169, 2345168, 2345167, 2345166, 2345165, 2345164, 2345163, 2345162, 2345161,
            2345160, 2345159, 2345158, 2345156, 2345155, 2345154, 2345153, 2345152, 2345151, 2345150, 2345149, 2345148,
            2345147, 2345146, 2345145, 2345144, 2345143, 2345142, 2345141, 2345140, 2345139, 2345138, 2345137, 2345136,
            2345135, 2345134, 2345133, 2345132, 2345131, 2345130, 2345129, 2345128, 2345127, 2345126, 2345125, 2345124,
            2345123, 2345122, 2345121, 2345120, 2345119, 2345118, 2345117, 2345116, 2345115, 2345114, 2345112, 2345111,
            2345110, 2345109, 2345108, 2345107, 2345106, 2345105, 2345104, 2345103, 2345102, 2345101, 2345100, 2345099,
            2345098, 2345097, 2345096, 2345095, 2345094, 2345093, 2345092, 2345091, 2345090, 2345089, 2345088, 2345087,
            2345086, 2345085, 2345084, 2345083, 2345082, 2345081, 2345080, 2345079, 2345078, 2345077, 2345076, 2345075,
            2345074, 2345073, 2345072, 2345071, 2345070, 2345069, 2345068, 2345067, 2345066, 2345065, 2345064, 2345063,
            2345062, 2345061, 2345060, 2345059, 2345058, 2345057, 2345056, 2345055, 2345054, 2345053, 2345052, 2345051,
            2345050, 2345048, 2345047, 2345046, 2345045, 2345044, 2345043, 2345042, 2345041, 2345040, 2345038, 2345037,
            2345036, 2345035, 2345034, 2345033, 2345032, 2345031, 2345030, 2345029, 2345027, 2345026, 2345025, 2345024,
            2345023, 2345022, 2345021, 2345019, 2345018, 2345017, 2345016, 2345015, 2345014, 2345013, 2345012, 2345011,
            2345010, 2345008, 2345007, 2345006, 2345005, 2345004, 2345003, 2345002, 2345001, 2345000, 2344999, 2344998,
            2344997, 2344996, 2344995, 2344994, 2344993, 2344992, 2344991, 2344990, 2344989, 2344987, 2344986, 2344985,
            2344984, 2344983, 2344982, 2344981, 2344980, 2344979, 2344978, 2344977, 2344974, 2344973, 2344972, 2344971,
            2344970, 2344969, 2344968, 2344967, 2344966, 2344965, 2344964, 2344963, 2344962, 2344961, 2344960, 2344959,
            2344958, 2344957, 2344956, 2344955, 2344954, 2344952, 2344951, 2344950, 2344949, 2344948, 2344947, 2344946,
            2344945, 2344944, 2344943, 2344942, 2344941, 2344940, 2344939, 2344938, 2344937, 2344936, 2344935, 2344934,
            2344933, 2344932, 2344931, 2344930, 2344929, 2344927, 2344926, 2344925, 2344924, 2344922, 2344921, 2344920,
            2344919, 2344918, 2344917, 2344916, 2344915, 2344914, 2344913, 2344912, 2344911, 2344910, 2344909, 2344908,
            2344905, 2344904, 2344902, 2344901, 2344900, 2344899, 2344898, 2344897, 2344896, 2344895, 2344894, 2344893,
            2344892, 2344891, 2344890, 2344889, 2344888, 2344887, 2344886, 2344885, 2344884, 2344883, 2344882, 2344881,
            2344880, 2344879, 2344878, 2344877, 2344876, 2344875, 2344874, 2344873, 2344872, 2344871, 2344870, 2344869,
            2344868, 2344867, 2344866, 2344865, 2344864, 2344863, 2344862, 2344861, 2344860, 2344859, 2344858, 2344857,
            2344856, 2344855, 2344854, 2344853, 2344852, 2344851, 2344850, 2344849, 2344848, 2344847, 2344846, 2344845,
            2344843, 2344842, 2344841, 2344840, 2344839, 2344838, 2344837, 2344836, 2344835, 2344834, 2344833, 2344832,
            2344831, 2344830, 2344829, 2344828, 2344827, 2344826, 2344825, 2344824, 2344823, 2344822, 2344821, 2344820,
            2344819, 2344818, 2344817, 2344816, 2344815, 2344814, 2344813, 2344812, 2344811, 2344810, 2344809, 2344808,
            2344807, 2344806, 2344805, 2344804, 2344803, 2344802, 2344801, 2344799, 2344798, 2344797, 2344796, 2344795,
            2344794, 2344793, 2344792, 2344791, 2344790, 2344789, 2344788, 2344787, 2344785, 2344784, 2344783, 2344782,
            2344781, 2344780, 2344779, 2344778, 2344777, 2344776, 2344775, 2344774, 2344773, 2344771, 2344770, 2344769,
            2344768, 2344767, 2344766, 2344764, 2344763, 2344762, 2344761, 2344760, 2344759, 2344758, 2344757, 2344756,
            2344755, 2344754, 2344753, 2344752, 2344751, 2344750, 2344749, 2344748, 2344747, 2344746, 2344745, 2344744,
            2344743, 2344742, 2344741, 2344740, 2344739, 2344738, 2344737, 2344736, 2344735, 2344734, 2344733, 2344731,
            2344730, 2344729, 2344728, 2344727, 2344726, 2344725, 2344724, 2344723, 2344722, 2344721, 2344720, 2344719,
            2344718, 2344717, 2344716, 2344715, 2344713, 2344712, 2344711, 2344710, 2344709, 2344708, 2344707, 2344706,
            2344705, 2344704, 2344702, 2344701, 2344700, 2344699, 2344698, 2344697, 2344696, 2344695, 2344694, 2344693,
            2344692, 2344691, 2344690, 2344689, 2344688, 2344687, 2344686, 2344685, 2344684, 2344683, 2344682, 2344681,
            2344680, 2344678, 2344677, 2344676, 2344675, 2344674, 2344673, 2344672, 2344671, 2344670, 2344669, 2344668,
            2344667, 2344666, 2344665, 2344664, 2344663, 2344662, 2344661, 2344660, 2344659, 2344658, 2344657, 2344656,
            2344655, 2344654, 2344653, 2344652, 2344651, 2344650, 2344649, 2344648, 2344647, 2344646, 2344645, 2344644,
            2344643, 2344642, 2344641, 2344640, 2344639, 2344638, 2344637, 2344636, 2344635, 2344634, 2344633, 2344632,
            2344631, 2344630, 2344629, 2344628, 2344627, 2344626, 2344625, 2344624, 2344623, 2344622, 2344621, 2344620,
            2344619, 2344618, 2344617, 2344616, 2344615, 2344614, 2344613, 2344612, 2344610, 2344609, 2344608, 2344607,
            2344606, 2344605, 2344604, 2344603, 2344602, 2344601, 2344600, 2344599, 2344598, 2344597, 2344596, 2344595,
            2344594, 2344593, 2344592, 2344591, 2344590, 2344589, 2344588, 2344587, 2344586, 2344585, 2344584, 2344583,
            2344582, 2344581, 2344579, 2344578, 2344577, 2344575, 2344574, 2344573, 2344572, 2344571, 2344569, 2344568,
            2344567, 2344565, 2344564, 2344563, 2344562, 2344561, 2344559, 2344558, 2344557, 2344556, 2344555, 2344554,
            2344553, 2344552, 2344551, 2344550, 2344549, 2344548, 2344547, 2344546, 2344545, 2344544, 2344543, 2344541,
            2344540, 2344539, 2344538, 2344536, 2344534, 2344533, 2344532, 2344531, 2344530, 2344529, 2344528, 2344527,
            2344526, 2344525, 2344524, 2344523, 2344521, 2344520, 2344518, 2344517, 2344516, 2344515, 2344514, 2344513,
            2344512, 2344511, 2344510, 2344509, 2344508, 2344507, 2344506, 2344505, 2344504, 2344503, 2344502, 2344501,
            2344500, 2344499, 2344498, 2344496, 2344495, 2344494, 2344493, 2344492, 2344491, 2344490, 2344488, 2344487,
            2344486, 2344485, 2344484, 2344483, 2344481, 2344479, 2344478, 2344477, 2344476, 2344475, 2344474, 2344473,
            2344472, 2344471, 2344470, 2344469, 2344468, 2344467, 2344466, 2344465, 2344463, 2344462, 2344461, 2344460,
            2344459, 2344458, 2344457, 2344456, 2344455, 2344454, 2344453, 2344452, 2344451, 2344450, 2344449, 2344448,
            2344447, 2344446, 2344445, 2344444, 2344443, 2344441, 2344440, 2344439, 2344438, 2344437, 2344436, 2344434,
            2344433, 2344432, 2344431, 2344430, 2344429, 2344428, 2344427, 2344426, 2344425, 2344423, 2344421, 2344420,
            2344419, 2344418, 2344417, 2344416, 2344415, 2344414, 2344413, 2344412, 2344411, 2344410, 2344409, 2344408,
            2344407, 2344405, 2344404, 2344403, 2344402, 2344401, 2344400, 2344399, 2344398, 2344397, 2344396, 2344395,
            2344394, 2344393, 2344392, 2344391, 2344390, 2344389, 2344388, 2344387, 2344386, 2344385, 2344384, 2344383,
            2344382, 2344381, 2344380, 2344379, 2344378, 2344377, 2344376, 2344374, 2344373, 2344371, 2344370, 2344369,
            2344368, 2344367, 2344366, 2344365, 2344364, 2344363, 2344362, 2344361, 2344360, 2344359, 2344358, 2344357,
            2344356, 2344355, 2344354, 2344353, 2344352, 2344351, 2344350, 2344349, 2344348, 2344346, 2344345, 2344344,
            2344343, 2344342, 2344341, 2344340, 2344339, 2344338, 2344337, 2344336, 2344334, 2344333, 2344332, 2344331,
            2344330, 2344329, 2344328, 2344327, 2344326, 2344325, 2344324, 2344323, 2344322, 2344321, 2344319, 2344318,
            2344317, 2344316, 2344315, 2344314, 2344313, 2344312, 2344311, 2344310, 2344309, 2344308, 2344307, 2344306,
            2344305, 2344304, 2344303, 2344302, 2344301, 2344300, 2344299, 2344298, 2344297, 2344296, 2344295, 2344294,
            2344293, 2344292, 2344291, 2344290, 2344289, 2344288, 2344287, 2344286, 2344285, 2344284, 2344283, 2344282,
            2344281, 2344280, 2344279, 2344278, 2344277, 2344276, 2344275, 2344274, 2344273, 2344272, 2344271, 2344270,
            2344268, 2344267, 2344266, 2344265, 2344264, 2344263, 2344262, 2344261, 2344260, 2344259, 2344258, 2344257,
            2344256, 2344255, 2344254, 2344253, 2344252, 2344251, 2344250, 2344249, 2344247, 2344246, 2344245, 2344244,
            2344243, 2344242, 2344241, 2344240, 2344239, 2344238, 2344237, 2344236, 2344235, 2344234, 2344231, 2344230,
            2344229, 2344228, 2344227, 2344226, 2344225, 2344224, 2344223, 2344222, 2344221, 2344220, 2344219, 2344218,
            2344217, 2344216, 2344215, 2344214, 2344213, 2344212, 2344211, 2344210, 2344209, 2344208, 2344207, 2344206,
            2344205, 2344204, 2344203, 2344202, 2344201, 2344200, 2344199, 2344198, 2344197, 2344195, 2344193, 2344192,
            2344191, 2344190, 2344189, 2344188, 2344187, 2344186, 2344185, 2344184, 2344182, 2344181, 2344180, 2344179,
            2344178, 2344177, 2344176, 2344175, 2344174, 2344173, 2344172, 2344171, 2344170, 2344169, 2344168, 2344167,
            2344166, 2344165, 2344164, 2344163, 2344162, 2344161, 2344160, 2344159, 2344158, 2344157, 2344156, 2344155,
            2344154, 2344153, 2344152, 2344150, 2344149, 2344148, 2344147, 2344146, 2344145, 2344144, 2344143, 2344142,
            2344141, 2344140, 2344139, 2344138, 2344137, 2344136, 2344135, 2344134, 2344133, 2344132, 2344130, 2344129,
            2344128, 2344127, 2344126, 2344125, 2344124, 2344123, 2344122, 2344121, 2344120, 2344119, 2344118, 2344117,
            2344116, 2344115, 2344114, 2344113, 2344112, 2344111, 2344110, 2344109, 2344108, 2344107, 2344106, 2344105,
            2344104, 2344102, 2344101, 2344100, 2344099, 2344098, 2344097, 2344096, 2344095, 2344094, 2344093, 2344092,
            2344091, 2344090, 2344089, 2344088, 2344087, 2344086, 2344085, 2344084, 2344083, 2344082, 2344081, 2344080,
            2344079, 2344078, 2344077, 2344076, 2344075, 2344073, 2344072, 2344071, 2344070, 2344069, 2344068, 2344067,
            2344066, 2344065, 2344064, 2344063, 2344062, 2344061, 2344060, 2344059, 2344058, 2344057, 2344056, 2344055,
            2344053, 2344052, 2344051, 2344050, 2344049, 2344048, 2344047, 2344045, 2344044, 2344043, 2344042, 2344041,
            2344040, 2344039, 2344038, 2344037, 2344036, 2344035, 2344034, 2344033, 2344032, 2344031, 2344030, 2344029,
            2344028, 2344027, 2344026, 2344024, 2344023, 2344022, 2344021, 2344020, 2344019, 2344018, 2344017, 2344016,
            2344015, 2344014, 2344013, 2344012, 2344011, 2344010, 2344009, 2344008, 2344007, 2344006, 2344004, 2344003,
            2344002, 2344001, 2344000, 2343999, 2343998, 2343997, 2343996, 2343994, 2343993, 2343992, 2343991, 2343990,
            2343989, 2343988, 2343987, 2343986, 2343985, 2343984, 2343983, 2343982, 2343980, 2343979, 2343977, 2343976,
            2343975, 2343974, 2343973, 2343972, 2343971, 2343970, 2343969, 2343968, 2343967, 2343966, 2343965, 2343964,
            2343963, 2343962, 2343961, 2343960, 2343959, 2343958, 2343957, 2343956, 2343955, 2343954, 2343953, 2343952,
            2343951, 2343950, 2343949, 2343948, 2343947, 2343946, 2343945, 2343944, 2343943, 2343942, 2343941, 2343940,
            2343939, 2343938, 2343937, 2343936, 2343935, 2343934, 2343933, 2343932, 2343931, 2343930, 2343929, 2343928,
            2343927, 2343926, 2343925, 2343924, 2343923, 2343922, 2343921, 2343920, 2343919, 2343918, 2343917, 2343915,
            2343914, 2343913, 2343911, 2343910, 2343909, 2343908, 2343907, 2343906, 2343905, 2343904, 2343903, 2343902,
            2343901, 2343900, 2343899, 2343898, 2343897, 2343896, 2343895, 2343894, 2343893, 2343892, 2343891, 2343890,
            2343889, 2343888, 2343887, 2343886, 2343885, 2343883, 2343882, 2343881, 2343880, 2343879, 2343878, 2343877,
            2343876, 2343875, 2343874, 2343873, 2343872, 2343871, 2343870, 2343869, 2343868, 2343867, 2343866, 2343865,
            2343864, 2343863, 2343862, 2343861, 2343859, 2343858, 2343857, 2343856, 2343855, 2343854, 2343853, 2343852,
            2343851, 2343850, 2343849, 2343848, 2343847, 2343846, 2343845, 2343844, 2343843, 2343842, 2343841, 2343840,
            2343839, 2343838, 2343837, 2343836, 2343835, 2343834, 2343832, 2343831, 2343830, 2343829, 2343828, 2343827,
            2343826, 2343825, 2343824, 2343823, 2343822, 2343821, 2343820, 2343819, 2343818, 2343816, 2343815, 2343814,
            2343813, 2343812, 2343811, 2343810, 2343809, 2343808, 2343807, 2343806, 2343805, 2343804, 2343803, 2343802,
            2343801, 2343800, 2343799, 2343798, 2343797, 2343796, 2343795, 2343794, 2343793, 2343792, 2343791, 2343790,
            2343789, 2343788, 2343787, 2343786, 2343785, 2343784, 2343783, 2343782, 2343781, 2343780, 2343779, 2343778,
            2343777, 2343776, 2343775, 2343774, 2343773, 2343772, 2343771, 2343770, 2343769, 2343768, 2343767, 2343766,
            2343765, 2343764, 2343763, 2343762, 2343761, 2343760, 2343759, 2343758, 2343757, 2343756, 2343755, 2343754,
            2343753, 2343752, 2343751, 2343750, 2343749, 2343748, 2343747, 2343746, 2343745, 2343744, 2343743, 2343742,
            2343740, 2343739, 2343738, 2343737, 2343736, 2343735, 2343734, 2343733, 2343732, 2343731, 2343730]


def url_exists(url):
    """
    Returns True if url exists and False otherwise.
    :param url:
    :return:
    """
    if url is None:
        return False

    return urllib.urlopen(url).code == 200


def downloadProbe(probe_full_path, probe_url):
    """
        Downloads the image and stores it locally.
        :param probe_url:
        :param probe_full_path:
        :return: A boolean indication if the probe is stored locally (success of the operation).
        """

    # First we start by checking if the file is already stored locally. If so there is no need try and access
    # remote resources.
    if os.path.isfile(probe_full_path):
        print('Probe image already exists in: ' + probe_full_path)
    elif url_exists(probe_url):
        print('Downloading probe: {} from {}'.format(probe_full_path, probe_url))

        try:
            num_try = 1
            timeout_interval = 30
            while num_try < 4:
                t = threading.Thread(name='urlretrieve thread', target=urllib.urlretrieve,
                                     args=(probe_url, probe_full_path))
                t.start()
                t.join(timeout=timeout_interval)
                if t.is_alive():
                    print('Time-out over {} seconds'.format(timeout_interval))
                    print(
                        'Num of try for downloading: {}. End Threading - the thread is still alive'.format(num_try))
                    num_try += 1
                    continue
                else:
                    break
            # After 3 tries stop downloading
            if num_try == 4:
                print('Time-out over {} seconds '.format(timeout_interval))
                print(
                    'End Threading - the thread is still alive. Stop downloading after 3 tries')
                return False

        except Exception as e:
            exception_full_msg = 'Failed to download image from {}\n{}'. \
                format(probe_url, e)
            print(exception_full_msg)
            return False
    else:
        print('Probe URL does not exists, image was not downloaded from: {}'.format(probe_url))
        return False

    return True
