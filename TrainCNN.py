from __future__ import print_function
import operator
from Data.VisualGenome.local import GetAllImageData, GetAllRegionDescriptions, GetSceneGraph, GetAllQAs
from keras_frcnn.Lib.PascalVocDataGenerator import PascalVocDataGenerator
# from keras_frcnn.Lib.Loss import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from keras_frcnn.Lib.VisualGenomeDataGenerator import VisualGenomeDataGenerator
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
from keras.layers import Input, AveragePooling2D, Flatten, Dense
from keras_frcnn.Lib.PascalVoc import PascalVoc
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from keras.models import Model
import cv2
import random
import matplotlib.pyplot as plt
from keras_frcnn.Utils.Utils import get_mask_from_object, create_folder, try_create_patch, VG_PATCH_PATH

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

NUM_EPOCHS = 50
# len(train_imgs)
TRAIN_SAMPLES_PER_EPOCH = 2000
# len(val_imgs)
NUM_VAL_SAMPLES = 500


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


def get_sorted_data(classes_count_file_name="80000_classes_count.p",
                    hierarchy_mapping_file_name="80000_class_mapping.p", entititis_file_name="80000_entities.p"):
    """
    This function his sorted the hierarchy_mapping and classes_count by the number of labels
    :param entititis_file_name: the full entities of *all* the dataset
    :param classes_count_file_name: classes count of *all* the dataset
    :param hierarchy_mapping_file_name: hierarchy_mapping of *all* the dataset
    :return:  a dict of classes_count (mapping between the class and its instances), a dict of hierarchy_mapping
    (mapping between the class and its object id), entities
    """

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
    classes_count_path = os.path.join(VisualGenome_PICKLES_PATH, CLASSES_COUNT_FILE)
    classes_mapping_path = os.path.join(VisualGenome_PICKLES_PATH, HIERARCHY_MAPPING)
    entities_path = os.path.join(VisualGenome_PICKLES_PATH, entititis_file_name)

    # Check if pickles are already created
    if os.path.isfile(classes_count_path) and os.path.isfile(classes_mapping_path) and os.path.isfile(entities_path):
        print(
            'Files are already exist {0}, {1} and {2}'.format(classes_count_path, classes_mapping_path, entities_path))
        classes_count = cPickle.load(file(classes_count_path, 'rb'))
        hierarchy_mapping = cPickle.load(file(classes_mapping_path, 'rb'))
        entities = np.array(cPickle.load(file(entities_path, 'rb')))
        return classes_count, hierarchy_mapping, entities

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
    random.shuffle(indices)
    train_imgs = entities[indices[:train_size]]
    test_imgs = entities[indices[train_size:train_size + test_size]]
    val_imgs = entities[indices[train_size + test_size:]]
    return train_imgs, test_imgs, val_imgs


if __name__ == '__main__':

    # Load class config
    config = Config()

    # Get Visual Genome Data
    classes_count, hierarchy_mapping, entities = get_sorted_data(classes_count_file_name="final_classes_count.p",
                                                                 hierarchy_mapping_file_name="final_class_mapping.p",
                                                                 entititis_file_name="entities_example.p")
    train_imgs, test_imgs, val_imgs = preprocessing_data(entities)

    number_of_classes = len(classes_count)

    # Get PascalVoc data
    train_imgs, val_imgs, hierarchy_mapping1, classes_count1 = create_data_pascal_voc(load=True)

    # Create a data generator for PascalVoc
    data_gen_train = PascalVocDataGenerator(data=train_imgs, hierarchy_mapping=hierarchy_mapping1,
                                            classes_count=classes_count1,
                                            config=config, backend=K.image_dim_ordering(), mode='train', batch_size=1)
    data_gen_val = PascalVocDataGenerator(data=val_imgs, hierarchy_mapping=hierarchy_mapping1,
                                          classes_count=classes_count1,
                                          config=config, backend=K.image_dim_ordering(), mode='test', batch_size=1)

    #
    # image_data = GetAllImageData(dataDir=DATA_PATH)
    # region_interest = GetAllRegionDescriptions(dataDir=DATA_PATH)
    # qas = GetAllQAs(dataDir=DATA_PATH)
    # tt = GetSceneGraph(1, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/", synsetFile=DATA_PATH + "synsets.json")

    # classes_count, hierarchy_mapping, entities = create_data_visual_genome(image_data)
    # exit()

    print("test")
    # Create a data generator for Visual Genome
    data_gen_train_vg = VisualGenomeDataGenerator(data=train_imgs, hierarchy_mapping=hierarchy_mapping,
                                                  classes_count=classes_count,
                                                  config=config, backend=K.image_dim_ordering(), mode='train',
                                                  batch_size=10)
    data_gen_test_vg = VisualGenomeDataGenerator(data=test_imgs, hierarchy_mapping=hierarchy_mapping,
                                                 classes_count=classes_count, config=config,
                                                 backend=K.image_dim_ordering(), mode='test', batch_size=5)
    # data_gen_train_vg.next()
    print("end test")

    # image_data = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "images_data.p"), "rb"))
    # qas = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "qas.p"), "rb"))
    # region_interest = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "region_interest.p"), "rb"))
    print('loading pickles')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    # Get back the ResNet50 base part of a ResNet50 network trained on MS-COCO
    # model_resnet50 = ResNet50(weights='imagenet', include_top=False)
    # model_resnet50.summary()

    img_input = Input(shape=(config.crop_width, config.crop_height, 3), name="image_input")

    net = ModelZoo()
    # Without Top
    model_resnet50 = net.resnet50_base(img_input, trainable=True)
    # Add AVG Pooling Layer
    # model_resnet50 = AveragePooling2D((7, 7), name='avg_pool')(model_resnet50)
    # Add the fully-connected layers
    model_resnet50 = Flatten(name='flatten')(model_resnet50)
    output_resnet50 = Dense(number_of_classes, activation='softmax', name='fc')(model_resnet50)

    # Define the model
    model = Model(input=img_input, output=output_resnet50)

    # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    model.summary()

    # Load pre-trained weights for ResNet50
    try:
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
                  loss='categorical_crossentropy')

    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
                 ModelCheckpoint(config.model_path, monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir="logs/", write_graph=False, write_images=True)]

    print('Starting training')
    history = model.fit_generator(data_gen_train, samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH, nb_epoch=NUM_EPOCHS,
                                  validation_data=data_gen_val, nb_val_samples=NUM_VAL_SAMPLES, callbacks=callbacks,
                                  max_q_size=10, nb_worker=1)

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
