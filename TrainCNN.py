from __future__ import print_function
import operator
from Data.VisualGenome.local import GetAllImageData, GetAllRegionDescriptions, GetSceneGraph, GetAllQAs
from Data.VisualGenome.models import ObjectMapping
from keras_frcnn.Lib.PascalVocDataGenerator import PascalVocDataGenerator
# from keras_frcnn.Lib.Loss import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from keras_frcnn.Lib.VisualGenomeDataGenerator import VisualGenomeDataGenerator, VisualGenomeDataGenerator_func
from keras_frcnn.Lib.Zoo import ModelZoo
from keras.applications.resnet50 import ResNet50
import random
import pprint
import os
import cPickle
import json
import numpy as np
from keras_frcnn.Lib.Config import Config
from keras.optimizers import Adam
from keras.layers import Input, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras_frcnn.Lib.PascalVoc import PascalVoc
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras import backend as K
from keras.models import Model
import cv2
import sys
import random
import matplotlib.pyplot as plt
from keras_frcnn.Utils.Utils import get_mask_from_object, create_folder, try_create_patch, VG_PATCH_PATH
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2

# If the allocation of training, validation and testing does not adds up to one
used_percent = TRAINING_PERCENT + VALIDATION_PERCENT + TESTING_PERCENT
if not used_percent == 1:
    error_msg = 'Data used percent (train + test + validation) is {0} and should be 1'.format(used_percent)
    print(error_msg)
    raise Exception(error_msg)

__author__ = 'roeih'

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
DATA_PATH = "Data/VisualGenome/data/"

NUM_EPOCHS = 128


def create_data_pascal_voc(load=False):
    """
    This function load pickles
    :param load: load field
    :return: train_imgs, val_imgs, class_mapping_test.p, classes_count
    """

    # When loading Pickle
    if load:
        class_mapping = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, CLASSES_MAPPING_FILE), "rb"))
        classes_count = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, CLASSES_COUNT_FILE), "rb"))
        train_imgs = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, TRAIN_IMGS_P), "rb"))
        val_imgs = cPickle.load(open(os.path.join(PascalVoc_PICKLES_PATH, VAL_IMGS_P), "rb"))
        print('loading pickles')
        return train_imgs, val_imgs, class_mapping, classes_count

    pascal_voc = PascalVoc()
    all_imgs, classes_count, class_mapping = pascal_voc.get_data("/home/roeih/PascalVoc/VOCdevkit",
                                                                 pascal_data=['VOC2007'])

    # Add background class
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    # Create json for the class for future use
    with open('classes.json', 'w') as class_data_json:
        json.dump(class_mapping, class_data_json)
    pprint.pprint(classes_count)

    # Shuffle the Data
    random.shuffle(all_imgs)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    return train_imgs, val_imgs, class_mapping, classes_count


def create_data_visual_genome(image_data):
    """
    This function creates or load pickles.
    hierarchy_mapping: dict with mapping between a class label and his object_id
    classes_count: dict with mapping between a class label and the number of his instances in the visual genome data
    :param image_data: image data
    :return: classes_count, hierarchy_mapping, entities
    """

    img_ids = [img.id for img in image_data]
    create_folder(VG_PATCH_PATH)
    classes_count = {}
    # Map between label to images
    hierarchy_mapping = {}

    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_COUNT_FILE)
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_MAPPING_FILE)
    entities_path = os.path.join(VisualGenome_PICKLES_PATH, ENTITIES_FILE)

    # Check if pickles are already created
    if os.path.isfile(classes_count_path) and os.path.isfile(classes_mapping_path) and os.path.isfile(entities_path):
        classes_count = cPickle.load(file(classes_count_path, 'rb'))
        hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
        entities = []
        # entities = cPickle.load(file(entities_path, 'rb'))
        return classes_count, hierarchy_mapping, entities

    # Create classes_count, hierarchy_mapping and entities
    entities = []
    # index for saving
    ind = 1
    print("Start creating pickle for VisualGenome Data with changes")
    for img_id in img_ids:
        try:
            entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
                                   synsetFile=DATA_PATH + "synsets.json")
            entities.append(entity)
            objects = entity.objects
            for object in objects:
                obj_id = object.id
                # if len(object.names) > 1:
                #     print("Two labels per object in img_id:{0}, object_id:{1}, labels:{2}".format(img_id, obj_id,
                #                                                                                   object.names))

                label = object.names[0]

                # Update the classes_count dict
                if label in classes_count:
                    # Check if label is already in dict
                    classes_count[label] += 1
                else:
                    # Init label in dict
                    classes_count[label] = 1

                # Update hierarchy_mapping dict
                if label not in hierarchy_mapping:
                    hierarchy_mapping[label] = obj_id

            # Printing Alerting
            if ind % 10000 == 0:
                save_pickles(classes_count, entities, hierarchy_mapping, iter=str(ind))
                # print("This is iteration number: {}".format(ind))
            # Updating index
            ind += 1
            print("This is iteration number: {}".format(img_id))

        except Exception as e:
            print("Problem with {0} in index: {1}".format(e, img_id))

    save_pickles(classes_count, entities, hierarchy_mapping, iter='final')
    return classes_count, hierarchy_mapping, entities
    # My future generator
    # for img_id in img_ids:
    #     img = cv2.imread(DATA_PATH + "/VG_100K_2/{}.jpg".format(img_id))
    #     entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
    #                            synsetFile=DATA_PATH + "synsets.json")
    #     objects = entity.objects
    #     for object in objects:
    #         obj_id = object.id
    #         mask = get_mask_from_object(object)
    #         patch_name = os.path.join(PATCH_PATH, "{0}_{1}.jpg".format(img_id, obj_id))
    #         print('mask')
    #         tryCreatePatch(img, mask, patch_name)
    #
    #     print("debug")


def save_pickles(classes_count, entities, hierarchy_mapping, iter=''):
    """
    This function save the pickles each iter
    :param classes_count: dict the classes count
    :param entities: dict with the entities
    :param hierarchy_mapping: dict with hierarchy mapping
    :param iter: string iteration number
    """
    # Save classes_count file
    classes_count_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + CLASSES_COUNT_FILE), 'wb')
    # Pickle classes_count
    cPickle.dump(classes_count, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Save hierarchy_mapping file
    hierarchy_mapping_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + CLASSES_MAPPING_FILE), 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    # Save entities list
    entities_file = file(os.path.join(VisualGenome_PICKLES_PATH, iter + '_' + ENTITIES_FILE), 'wb')
    # Pickle entities
    cPickle.dump(entities, entities_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    entities_file.close()


def load_pickles(classes_mapping_path, classes_count_path, entities_path, ):
    """
    This function save the pickles each iter
    :param classes_count_path: classes_count file name
    :param entities_path: entities file name
    :param classes_mapping_path: hierarchy_mapping file name
    :return classes_count, hierarchy_mapping and entities
    """
    classes_count = cPickle.load(file(classes_count_path, 'rb'))
    hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
    entities = cPickle.load(file(entities_path, 'rb'))
    return classes_count, hierarchy_mapping, entities


def get_sorted_data(classes_count_file_name="final_classes_count.p",
                    hierarchy_mapping_file_name="final_class_mapping.p", entititis_file_name="final_entities.p"):
    """
    This function his sorted the hierarchy_mapping and classes_count by the number of labels
    :param entitis_file_name: the full entities of *all* the dataset
    :param classes_count_file_name: classes count of *all* the dataset
    :param hierarchy_mapping_file_name: hierarchy_mapping of *all* the dataset
    :return:  a dict of classes_count (mapping between the class and its instances), a dict of hierarchy_mapping
    (mapping between the class and its object id), entities
    """

    # Check if pickles are already created
    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_COUNT_FILE)
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, HIERARCHY_MAPPING)
    entities_path = os.path.join(VisualGenome_PICKLES_PATH, entititis_file_name)

    if os.path.isfile(classes_count_path) and os.path.isfile(classes_mapping_path) and os.path.isfile(entities_path):
        print(
            'Files are already exist {0}, {1} and {2}'.format(classes_count_path, classes_mapping_path, entities_path))
        classes_count = cPickle.load(file(classes_count_path, 'rb'))
        hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
        entities = np.array(cPickle.load(file(entities_path, 'rb')))
        return classes_count, hierarchy_mapping, entities

    # Sort and pre-process the 3 pickle files

    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, classes_count_file_name)
    # Load the frequency of labels
    classes_count = cPickle.load(file(classes_count_path, 'rb'))
    # Sort classes_count by value
    sorted_classes_count = sorted(classes_count.items(), key=operator.itemgetter(1), reverse=True)
    # Get the most frequent 150 labels
    top_sorted_class = sorted_classes_count[:NOF_LABELS]
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, hierarchy_mapping_file_name)
    # Load the full hierarchy_mapping
    hierarchy_mapping_full = cPickle.load(file(classes_mapping_path, 'rb'))

    # Create a new hierarchy_mapping for top labels
    hierarchy_mapping = {}
    top_sorted_class_keys = [classes[0] for classes in top_sorted_class]
    for key in hierarchy_mapping_full:
        if key in top_sorted_class_keys:
            hierarchy_mapping[key] = hierarchy_mapping_full[key]

            # Save hierarchy_mapping file for only the top labels
    hierarchy_mapping_file = file(classes_mapping_path, 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(hierarchy_mapping, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
    # Save classes_count file for only the top labels
    classes_count_file = file(classes_count_path, 'wb')
    # Pickle hierarchy_mapping
    cPickle.dump(dict(top_sorted_class), classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Load entities pickle
    entities = np.array(cPickle.load(file(entities_path, 'rb')))
    return classes_count, hierarchy_mapping, entities


def preprocessing_data(entities):
    """
    This function splits the data for train and test dataset
    :param entities:
    :return: list of entities of train and test data
    """
    number_of_samples = len(entities)
    train_size = int(number_of_samples * TRAINING_PERCENT)
    test_size = int(number_of_samples * TESTING_PERCENT)
    validation_size = number_of_samples - (train_size + test_size)

    if not train_size + test_size + validation_size == number_of_samples:
        error_msg = 'Data size of (train + test + validation) is {0} and should be number of labels: {1}'.format(
            train_size + test_size + validation_size, number_of_samples)
        print(error_msg)
        raise Exception(error_msg)

    # Create a numpy array of indices of the data
    indices = np.arange(len(entities))
    # Shuffle the indices of the data
    # todo: must returned the shuffle
    random.shuffle(indices)

    # Get the train + test + val dataset
    train_imgs = entities[indices[:train_size]]
    test_imgs = entities[indices[train_size:train_size + test_size]]
    val_imgs = entities[indices[train_size + test_size:]]

    # Take the round number of each dataset per the number of epochs
    num_of_samples_per_train_updated = len(train_imgs) / NUM_EPOCHS * NUM_EPOCHS
    train_imgs = train_imgs[:num_of_samples_per_train_updated]
    num_of_samples_per_test_updated = len(test_imgs) / NUM_EPOCHS * NUM_EPOCHS
    test_imgs = test_imgs[:num_of_samples_per_test_updated]
    num_of_samples_per_val_updated = len(val_imgs) / NUM_EPOCHS * NUM_EPOCHS
    val_imgs = val_imgs[:num_of_samples_per_val_updated]

    return train_imgs, test_imgs, val_imgs


def process_objects(img_data, hierarchy_mapping, object_file_name='objects.p'):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param object_file_name: object pickle file name
    :param img_data: list of entities files
    :param hierarchy_mapping: dict of hierarchy_mapping
    :return: list of ObjectMapping
    """

    # Check if pickles are already created
    objects_path = os.path.join(VisualGenome_PICKLES_PATH, object_file_name)

    if os.path.isfile(objects_path):
        print('File is already exist {0}'.format(objects_path))
        objects = cPickle.load(file(objects_path, 'rb'))
        return objects

    # Get the whole objects from entities
    objects_lst = []
    correct_labels = hierarchy_mapping.keys()
    idx = 0
    for img in img_data:

        # Get the url image
        url = img.image.url
        # Get the objects per image
        objects = img.objects
        for object in objects:

            # Get the lable of object
            label = object.names[0]

            # Check if it is a correct label
            if label not in correct_labels:
                continue

            new_object_mapping = ObjectMapping(object.id, object.x, object.y, object.width, object.height, object.names,
                                               object.synsets, url)
            # Append the new objectMapping to objects_lst
            objects_lst.append(new_object_mapping)

        idx += 1
        print("Finished img: {}".format(idx))

    # Save the objects files to the disk
    objects_file = file(objects_path, 'wb')
    # Pickle objects_lst
    objects_array = np.array(objects_lst)
    cPickle.dump(objects_array, objects_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    objects_file.close()
    return objects_array


def get_new_hierarchy_mapping(hierarchy_mapping):
    """
    This function counts new hierarchy mapping from index 0 to number of classes
    :param hierarchy_mapping: a dict with mapping between label string to an object id from visual genome dataset
    :return: new dict with mapping between label string and a new count from 0 to number of classes
    """

    ind = 0
    new_hierarchy_mapping = {}
    for label in hierarchy_mapping.keys():
        new_hierarchy_mapping[label] = ind
        ind += 1

    return new_hierarchy_mapping


def get_classes_mapping_and_hierarchy_mapping_by_objects(objects):
    """
    This function creates classes_mapping and hierarchy_mapping by objects
    :param objects: list of objects
    :return: dict of classes_mapping and hierarchy_mapping
    """
    classes_count_per_objects = {}
    hierarchy_mapping_per_objects = {}
    new_obj_id = 1
    for object in objects:
        # Get the lable of object
        label = object.names[0]

        # Update the classes_count dict
        if label in classes_count_per_objects:
            # Check if label is already in dict
            classes_count_per_objects[label] += 1
        else:
            # Init label in dict
            classes_count_per_objects[label] = 1

        # Update hierarchy_mapping dict
        if label not in hierarchy_mapping_per_objects:
            hierarchy_mapping_per_objects[label] = new_obj_id
            new_obj_id += 1
    return classes_count_per_objects, hierarchy_mapping_per_objects


if __name__ == '__main__':

    # Get argument
    if len(sys.argv) < 2:
        # Default GPU number
        gpu_num = 0
    else:
        # Get the GPU number from the user
        gpu_num = sys.argv[1]

    # Load class config
    config = Config(gpu_num)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)

    # Define tensorflow use only the amount of memory required for the process
    # config_tf = tf.ConfigProto()
    # config_tf.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config_tf))

    classes_count, hierarchy_mapping, entities = get_sorted_data(classes_count_file_name="final_classes_count.p",
                                                                 hierarchy_mapping_file_name="final_class_mapping.p",
                                                                 entititis_file_name="entities_example.p")

    # Get Visual Genome Data objects
    objects = process_objects(entities, hierarchy_mapping, object_file_name="objects.p")

    # Only for debug
    classes_count_per_objects, hierarchy_mapping_per_objects = get_classes_mapping_and_hierarchy_mapping_by_objects(objects)

    train_imgs, test_imgs, val_imgs = preprocessing_data(objects)

    # Set the number of classes
    number_of_classes = len(classes_count)

    # Get PascalVoc data
    train_imgs1, val_imgs1, hierarchy_mapping1, classes_count1 = create_data_pascal_voc(load=True)

    # Create a data generator for PascalVoc
    data_gen_train = PascalVocDataGenerator(data=train_imgs1, hierarchy_mapping=hierarchy_mapping1,
                                            classes_count=classes_count1,
                                            config=config, backend=K.image_dim_ordering(), mode='train', batch_size=1)
    data_gen_val = PascalVocDataGenerator(data=val_imgs1, hierarchy_mapping=hierarchy_mapping1,
                                          classes_count=classes_count1,
                                          config=config, backend=K.image_dim_ordering(), mode='test', batch_size=1)

    # Create a data generator for VisualGenome
    data_gen_train_vg = VisualGenomeDataGenerator_func(data=train_imgs, hierarchy_mapping=new_hierarchy_mapping,
                                                       config=config, mode='train')
    data_gen_test_vg = VisualGenomeDataGenerator_func(data=test_imgs, hierarchy_mapping=new_hierarchy_mapping,
                                                      config=config, mode='test')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (config.crop_height, config.crop_width, 3)

    img_input = Input(shape=input_shape_img, name="image_input")

    # Define ResNet50 model Without Top
    net = ModelZoo()
    model_resnet50 = net.resnet50_base(img_input, trainable=True)
    model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
    output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
        model_resnet50)

    # Define the model
    model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
    # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    model.summary()

    # Get back the ResNet50 base part of a ResNet50 network trained on MS-COCO
    # model = ResNet50(weights=None, include_top=True, classes=number_of_classes)
    # model.summary()

    # Load pre-trained weights for ResNet50
    try:
        if config.load_weights:
            print('loading weights from {}'.format(config.base_net_weights))
            model.load_weights(config.base_net_weights, by_name=True)
    except Exception as e:
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        raise Exception(e)

    optimizer = Adam(1e-6)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ModelCheckpoint(config.model_path, monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir="logs", write_graph=True, write_images=True),
                 CSVLogger('training.log', separator=',', append=False)]

    print('Starting training')
    history = model.fit_generator(data_gen_train_vg, steps_per_epoch=len(train_imgs) / NUM_EPOCHS, epochs=NUM_EPOCHS,
                                  validation_data=data_gen_test_vg, validation_steps=len(test_imgs) / NUM_EPOCHS,
                                  callbacks=callbacks, max_q_size=1, workers=1)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# from keras.layers import Input, Flatten, Dense
# from keras.models import Model
# import numpy as np
#
# # Load class config
# config = Config()
#
# # Get back the ResNet50 base part of a ResNet50 network trained on MS-COCO
# model_resnet50 = ResNet50(weights='imagenet', include_top=False)
# model_resnet50.summary()
#
# img_input = Input(shape=(200, 200, 3), name="image_input")
#
# net = ModelZoo()
# # Without Top
# model_resnet50 = net.resnet50_base(img_input, trainable=True)
# # Add AVG Pooling Layer
# model_resnet50 = AveragePooling2D((7, 7), name='avg_pool')(model_resnet50)
# # Add the fully-connected layers
# model_resnet50 = Flatten(name='flatten')(model_resnet50)
# output_resnet50 = Dense(classes, activation='softmax', name='fc')(model_resnet50)
#
# # Define the model
# model = Model(input=img_input, output=output_resnet50)
#
# # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
# model.summary()
#
# # Load pre-trained weights for ResNet50
# try:
#     print('loading weights from {}'.format(config.base_net_weights))
#     model.load_weights(config.base_net_weights, by_name=True)
# except Exception as e:
#     print(e)
#     print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
#         'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
#         'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     ))

# # Use the generated model
# output_resnet50 = model_resnet50(input)
#
# # Create your own model
# my_model = Model(input=img_input, output=output_resnet50)
#
# # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# my_model.summary()
#
# # Then training with your data !
#
# # create graph of your new model
# head_model = Model(input=base_model.input, output=predictions)
#
# # compile the model
# head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# head_model.summary()
#
# # train your model on data
# head_model.fit(x, y, batch_size=batch_size, verbose=1)
