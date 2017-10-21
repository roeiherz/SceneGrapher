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

PROJECT_ROOT = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/"
# PROJECT_ROOT = "/home/roeih/SceneGrapher/"
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


def get_img(url, download=False):
    """
    This function read image from VisualGenome dataset as url and returns the image from local hard-driver
    :param download: A flag if we want to download the image
    :param url: url of the image
    :return: the image
    """
    try:
        path_lst = url.split('/')
        img_path = os.path.join(PROJECT_ROOT, VG_DATA_PATH, path_lst[-2], path_lst[-1])

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
            "https://cs.stanford.edu/people/rak248/VG_100K/2338230.jpg"]


def get_mini_entities_img_ids():
    """
    This function returns the 1000 image_ids by FEI-FEI paper
    :return:
    """
    img_ids = [2339172, 2339171, 2339170, 2339169, 2339168, 2339167, 2339166,
               2339165, 2339164, 2339163, 2339162, 2339161, 2339160, 2339159,
               2339158, 2339157, 2339156, 2339155, 2339154, 2339153, 2339152,
               2339151, 2339150, 2339149, 2339148, 2339147, 2339146, 2339145,
               2339144, 2339143, 2339142, 2339141, 2339140, 2339139, 2339138,
               2339137, 2339136, 2339135, 2339134, 2339133, 2339132, 2339131,
               2339130, 2339129, 2339127, 2339126, 2339125, 2339124, 2339123,
               2339122, 2339121, 2339120, 2339119, 2339118, 2339117, 2339116,
               2339115, 2339114, 2339113, 2339112, 2339110, 2339109, 2339108,
               2339107, 2339106, 2339105, 2339104, 2339103, 2339102, 2339101,
               2339100, 2339099, 2339098, 2339097, 2339096, 2339095, 2339094,
               2339093, 2339092, 2339091, 2339090, 2339089, 2339088, 2339087,
               2339086, 2339085, 2339084, 2339083, 2339082, 2339081, 2339080,
               2339079, 2339078, 2339077, 2339076, 2339075, 2339074, 2339073,
               2339072, 2339071, 2339070, 2339069, 2339068, 2339067, 2339066,
               2339065, 2339064, 2339063, 2339062, 2339061, 2339060, 2339059,
               2339058, 2339057, 2339056, 2339055, 2339054, 2339053, 2339052,
               2339051, 2339050, 2339049, 2339048, 2339047, 2339046, 2339045,
               2339044, 2339043, 2339042, 2339041, 2339040, 2339039, 2339037,
               2339036, 2339034, 2339033, 2339031, 2339030, 2339029, 2339028,
               2339027, 2339026, 2339025, 2339024, 2339023, 2339022, 2339021,
               2339020, 2339019, 2339018, 2339017, 2339016, 2339015, 2339014,
               2339013, 2339012, 2339011, 2339010, 2339009, 2339008, 2339007,
               2339006, 2339005, 2339004, 2339003, 2339002, 2339001, 2339000,
               2338999, 2338998, 2338997, 2338996, 2338995, 2338994, 2338993,
               2338992, 2338991, 2338990, 2338989, 2338988, 2338987, 2338986,
               2338985, 2338984, 2338983, 2338982, 2338981, 2338980, 2338979,
               2338978, 2338977, 2338976, 2338975, 2338974, 2338973, 2338972,
               2338971, 2338970, 2338969, 2338968, 2338967, 2338966, 2338965,
               2338964, 2338963, 2338962, 2338961, 2338960, 2338959, 2338958,
               2338957, 2338956, 2338955, 2338954, 2338953, 2338952, 2338951,
               2338950, 2338949, 2338948, 2338947, 2338946, 2338945, 2338944,
               2338943, 2338942, 2338941, 2338940, 2338939, 2338938, 2338937,
               2338936, 2338935, 2338934, 2338933, 2338932, 2338931, 2338930,
               2338929, 2338928, 2338927, 2338925, 2338924, 2338923, 2338922,
               2338921, 2338920, 2338919, 2338918, 2338917, 2338916, 2338915,
               2338914, 2338913, 2338912, 2338911, 2338910, 2338909, 2338908,
               2338907, 2338906, 2338905, 2338904, 2338903, 2338902, 2338901,
               2338900, 2338899, 2338898, 2338897, 2338896, 2338895, 2338894,
               2338893, 2338891, 2338890, 2338889, 2338888, 2338887, 2338886,
               2338885, 2338882, 2338881, 2338880, 2338879, 2338878, 2338877,
               2338876, 2338875, 2338874, 2338873, 2338872, 2338871, 2338870,
               2338869, 2338868, 2338867, 2338866, 2338865, 2338864, 2338863,
               2338862, 2338861, 2338860, 2338859, 2338858, 2338857, 2338856,
               2338855, 2338854, 2338853, 2338852, 2338851, 2338850, 2338849,
               2338848, 2338847, 2338846, 2338845, 2338844, 2338843, 2338842,
               2338841, 2338840, 2338839, 2338838, 2338837, 2338836, 2338835,
               2338834, 2338833, 2338831, 2338830, 2338829, 2338828, 2338827,
               2338826, 2338825, 2338824, 2338822, 2338821, 2338820, 2338819,
               2338817, 2338816, 2338815, 2338814, 2338812, 2338811, 2338810,
               2338809, 2338808, 2338807, 2338806, 2338805, 2338804, 2338803,
               2338802, 2338801, 2338799, 2338798, 2338797, 2338796, 2338795,
               2338794, 2338793, 2338792, 2338790, 2338789, 2338788, 2338787,
               2338786, 2338785, 2338784, 2338783, 2338782, 2338781, 2338780,
               2338779, 2338778, 2338777, 2338776, 2338775, 2338774, 2338773,
               2338772, 2338771, 2338770, 2338768, 2338767, 2338766, 2338765,
               2338763, 2338762, 2338761, 2338760, 2338759, 2338757, 2338756,
               2338755, 2338754, 2338753, 2338752, 2338751, 2338750, 2338749,
               2338748, 2338746, 2338745, 2338744, 2338743, 2338742, 2338740,
               2338739, 2338738, 2338737, 2338736, 2338735, 2338734, 2338733,
               2338732, 2338731, 2338730, 2338729, 2338728, 2338727, 2338726,
               2338725, 2338724, 2338723, 2338722, 2338721, 2338720, 2338719,
               2338718, 2338717, 2338716, 2338715, 2338714, 2338713, 2338712,
               2338711, 2338710, 2338709, 2338708, 2338707, 2338706, 2338705,
               2338704, 2338703, 2338702, 2338701, 2338700, 2338699, 2338698,
               2338697, 2338696, 2338695, 2338694, 2338693, 2338692, 2338691,
               2338690, 2338689, 2338688, 2338687, 2338686, 2338685, 2338684,
               2338683, 2338682, 2338681, 2338680, 2338679, 2338678, 2338677,
               2338676, 2338675, 2338674, 2338673, 2338672, 2338671, 2338670,
               2338669, 2338668, 2338666, 2338665, 2338664, 2338663, 2338662,
               2338661, 2338660, 2338659, 2338658, 2338657, 2338656, 2338655,
               2338653, 2338652, 2338651, 2338650, 2338649, 2338648, 2338647,
               2338646, 2338645, 2338644, 2338643, 2338642, 2338641, 2338640,
               2338639, 2338638, 2338637, 2338636, 2338634, 2338633, 2338632,
               2338631, 2338630, 2338629, 2338628, 2338627, 2338626, 2338625,
               2338624, 2338621, 2338620, 2338619, 2338618, 2338617, 2338616,
               2338615, 2338614, 2338613, 2338612, 2338611, 2338610, 2338609,
               2338608, 2338607, 2338606, 2338605, 2338603, 2338602, 2338601,
               2338600, 2338599, 2338598, 2338597, 2338596, 2338595, 2338594,
               2338593, 2338592, 2338591, 2338590, 2338589, 2338588, 2338587,
               2338586, 2338585, 2338584, 2338583, 2338582, 2338581, 2338580,
               2338578, 2338577, 2338575, 2338574, 2338573, 2338572, 2338571,
               2338570, 2338569, 2338568, 2338567, 2338566, 2338565, 2338564,
               2338563, 2338562, 2338561, 2338560, 2338559, 2338558, 2338557,
               2338556, 2338555, 2338554, 2338553, 2338552, 2338551, 2338550,
               2338549, 2338548, 2338547, 2338546, 2338545, 2338544, 2338543,
               2338542, 2338541, 2338540, 2338539, 2338538, 2338537, 2338536,
               2338535, 2338534, 2338533, 2338532, 2338531, 2338530, 2338529,
               2338528, 2338527, 2338526, 2338525, 2338524, 2338523, 2338522,
               2338521, 2338520, 2338519, 2338518, 2338517, 2338516, 2338515,
               2338514, 2338513, 2338512, 2338511, 2338510, 2338509, 2338508,
               2338507, 2338506, 2338505, 2338504, 2338503, 2338502, 2338501,
               2338500, 2338499, 2338498, 2338496, 2338495, 2338494, 2338493,
               2338492, 2338491, 2338490, 2338488, 2338487, 2338486, 2338485,
               2338484, 2338483, 2338482, 2338481, 2338480, 2338479, 2338478,
               2338477, 2338476, 2338475, 2338474, 2338473, 2338472, 2338471,
               2338470, 2338469, 2338468, 2338467, 2338466, 2338465, 2338464,
               2338463, 2338462, 2338461, 2338460, 2338459, 2338458, 2338457,
               2338456, 2338455, 2338454, 2338453, 2338452, 2338451, 2338450,
               2338449, 2338448, 2338447, 2338446, 2338445, 2338444, 2338443,
               2338442, 2338441, 2338440, 2338439, 2338438, 2338437, 2338436,
               2338435, 2338434, 2338433, 2338432, 2338431, 2338430, 2338429,
               2338428, 2338427, 2338425, 2338424, 2338423, 2338422, 2338421,
               2338420, 2338419, 2338418, 2338417, 2338416, 2338415, 2338414,
               2338413, 2338412, 2338411, 2338410, 2338409, 2338408, 2338407,
               2338406, 2338405, 2338404, 2338402, 2338401, 2338400, 2338399,
               2338398, 2338396, 2338395, 2338394, 2338393, 2338392, 2338391,
               2338390, 2338389, 2338388, 2338386, 2338385, 2338384, 2338383,
               2338382, 2338381, 2338380, 2338379, 2338378, 2338377, 2338376,
               2338375, 2338374, 2338373, 2338372, 2338371, 2338370, 2338369,
               2338368, 2338367, 2338366, 2338365, 2338364, 2338363, 2338362,
               2338361, 2338360, 2338359, 2338357, 2338356, 2338355, 2338354,
               2338353, 2338350, 2338349, 2338348, 2338347, 2338346, 2338345,
               2338344, 2338343, 2338342, 2338341, 2338340, 2338339, 2338338,
               2338337, 2338336, 2338335, 2338334, 2338333, 2338332, 2338331,
               2338330, 2338329, 2338328, 2338326, 2338325, 2338324, 2338323,
               2338322, 2338321, 2338320, 2338319, 2338318, 2338317, 2338316,
               2338315, 2338314, 2338313, 2338312, 2338311, 2338310, 2338309,
               2338308, 2338307, 2338306, 2338305, 2338304, 2338303, 2338302,
               2338301, 2338300, 2338299, 2338298, 2338297, 2338296, 2338295,
               2338294, 2338293, 2338292, 2338291, 2338290, 2338289, 2338288,
               2338287, 2338286, 2338285, 2338284, 2338283, 2338282, 2338281,
               2338280, 2338279, 2338278, 2338277, 2338276, 2338275, 2338274,
               2338273, 2338272, 2338271, 2338270, 2338269, 2338268, 2338267,
               2338266, 2338265, 2338264, 2338263, 2338262, 2338261, 2338260,
               2338259, 2338258, 2338257, 2338256, 2338255, 2338254, 2338253,
               2338252, 2338251, 2338250, 2338249, 2338247, 2338246, 2338245,
               2338244, 2338242, 2338241, 2338240, 2338239, 2338238, 2338237,
               2338236, 2338235, 2338234, 2338233, 2338232, 2338231, 2338230,
               2338229, 2338228, 2338227, 2338226, 2338225, 2338224, 2338223,
               2338222, 2338221, 2338220, 2338219, 2338218, 2338217, 2338216,
               2338215, 2338214, 2338213, 2338212, 2338211, 2338210, 2338209,
               2338208, 2338207, 2338206, 2338205, 2338204, 2338202, 2338201,
               2338200, 2338199, 2338198, 2338197, 2338196, 2338195, 2338194,
               2338193, 2338192, 2338191, 2338190, 2338189, 2338188, 2338187,
               2338186, 2338185, 2338184, 2338183, 2338181, 2338180, 2338178,
               2338177, 2338176, 2338175, 2338174, 2338173, 2338172, 2338171,
               2338170, 2338169, 2338168, 2338167, 2338166, 2338165, 2338164,
               2338163, 2338162, 2338161, 2338160, 2338159, 2338158, 2338157,
               2338156, 2338155, 2338154, 2338153, 2338152, 2338151, 2338150,
               2338149, 2338148, 2338147, 2338146, 2338145, 2338144, 2338143,
               2338142, 2338141, 2338140, 2338139, 2338138, 2338137, 2338136,
               2338135, 2338134, 2338133, 2338131, 2338130, 2338129]

    return img_ids


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
