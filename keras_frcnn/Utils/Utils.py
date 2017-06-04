import numpy
import cv2
import math
import os
import time
import matplotlib.pyplot as plt
from keras.engine import Model
from keras.layers import Dense

__author__ = 'roeih'


FILE_EXISTS_ERROR = (17, 'File exists')
PROJECT_ROOT = "/specific/netapp5_2/gamir/DER-Roei/SceneGrapher/"
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
VisualGenome_DATASETS_PICKLES_PATH = "keras_frcnn/PicklesDataset"
VG_VisualModule_PICKLES_PATH = "VisualModule/Data/VisualGenome"
MINI_VG_DATADET_PATH = "/home/roeih/VisualGenome/vg"
DATA_PATH = "Data/VisualGenome/data/"
TRAIN_DATA_SET = "train_set.p"
TEST_DATA_SET = "test_set.p"
VALIDATION_DATA_SET = "validation_set.p"
MINI_IMDB = "mini_imdb_1024.h5"
TRAINING_OBJECTS_CNN_PATH = "Training/TrainingObjectsCNN"
TRAINING_PREDICATE_CNN_PATH = "Training/TrainingPredicatesCNN"
WEIGHTS_NAME = 'model_vg_resnet50.hdf5'


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


def get_img(url):
    """
    This function read image from VisualGenome dataset as url and returns the image from local hard-driver
    :param url: url of the image
    :return: the image
    """
    try:
        path_lst = url.split('/')
        img_path = os.path.join(PROJECT_ROOT, VG_DATA_PATH, path_lst[-2], path_lst[-1])

        if not os.path.isfile(img_path):
            print("Error. Image path was not found")

        img = cv2.imread(img_path)

    except Exception as e:
        print(str(e))
        return None

    return img


def get_sorting_url():
    """
    This function sorting bad urls
    :return: a list of bad urls
    """
    lst = ["https://cs.stanford.edu/people/rak248/VG_100K/2321818.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K/2334844.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K_2/3807.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K_2/2410658.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K/2374264.jpg"]

    return lst