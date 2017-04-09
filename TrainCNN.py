from __future__ import print_function

from Data.VisualGenome.local import GetAllImageData, GetAllRegionDescriptions, GetSceneGraph, GetAllQAs
from keras_frcnn.Lib.DataGenerator import DataGenerator
from keras_frcnn.Lib.Loss import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
from keras_frcnn.Lib.Zoo import ModelZoo
from keras.applications.resnet50 import ResNet50
import random
import pprint
import os
import cPickle
import json
from keras_frcnn.Lib.Config import Config
from keras.optimizers import Adam
from keras.layers import Input, AveragePooling2D, Flatten, Dense
from keras_frcnn.Lib.PascalVoc import PascalVoc
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import Model
import cv2

from keras_frcnn.Utils.Utils import get_mask_from_object, create_folder, tryCreatePatch

__author__ = 'roeih'

SAVE_PATH = "keras_frcnn/Data"
PATCH_PATH = "Data/VisualGenome/Patches"

NUM_EPOCHS = 50
# len(train_imgs)
TRAIN_SAMPLES_PER_EPOCH = 2000
# len(val_imgs)
NUM_VAL_SAMPLES = 500


def create_data_pascal_voc(load=False):
    """
    This function load pickles
    :param load: load field
    :return: train_imgs, val_imgs, class_mapping.p, classes_count
    """

    # When loading Pickle
    if load:
        class_mapping = cPickle.load(open(os.path.join(SAVE_PATH, "class_mapping.p"), "rb"))
        classes_count = cPickle.load(open(os.path.join(SAVE_PATH, "classes_count.p"), "rb"))
        train_imgs = cPickle.load(open(os.path.join(SAVE_PATH, "train_imgs.p"), "rb"))
        val_imgs = cPickle.load(open(os.path.join(SAVE_PATH, "val_imgs.p"), "rb"))
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


def create_data_visual_genome():
    """
    This function load pickles
    :param load: load field
    :return: train_imgs, val_imgs, class_mapping.p, classes_count
    """

    img_ids = [img.id for img in image_data]
    create_folder(PATCH_PATH)
    classes_count = {}
    # Map between label to images
    hierarchy_mapping = {}

    # Get classes_count and hierarchy_mapping for mapping
    for img_id in img_ids:
        entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
                               synsetFile=DATA_PATH + "synsets.json")
        objects = entity.objects
        for object in objects:
            obj_id = object.id
            if len(object.names) > 1:
                print("Two labels per object in img_id:{0}, object_id:{1}".format(img_id, obj_id))

            label = object.names[0]

            # Update the dict
            if label in classes_count:
                # Check if label is already in dict
                classes_count[label] +=  1
            else:
                # Init label in dict
                classes_count[label] = 1
            print("debug")

    for img_id in img_ids:
        img = cv2.imread(DATA_PATH + "/VG_100K_2/{}.jpg".format(img_id))
        entity = GetSceneGraph(img_id, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/",
                               synsetFile=DATA_PATH + "synsets.json")
        objects = entity.objects
        for object in objects:
            obj_id = object.id
            mask = get_mask_from_object(object)
            patch_name = os.path.join(PATCH_PATH, "{0}_{1}.jpg".format(img_id, obj_id))
            print('mask')
            tryCreatePatch(img, mask, patch_name)

        print("debug")


if __name__ == '__main__':

    # Load class config
    config = Config()

    train_imgs, val_imgs, hierarchy_mapping, classes_count = create_data_pascal_voc(load=True)
    data_gen_train = DataGenerator(data=train_imgs, hierarchy_mapping=hierarchy_mapping, classes_count=classes_count,
                                   config=config, backend=K.image_dim_ordering(), mode='train', batch_size=1)
    data_gen_val = DataGenerator(data=val_imgs, hierarchy_mapping=hierarchy_mapping, classes_count=classes_count,
                                 config=config, backend=K.image_dim_ordering(), mode='test', batch_size=1)

    print("test")
    DATA_PATH = "Data/VisualGenome/data/"
    image_data = GetAllImageData(dataDir=DATA_PATH)
    # region_interest = GetAllRegionDescriptions(dataDir=DATA_PATH)
    # qas = GetAllQAs(dataDir=DATA_PATH)
    tt = GetSceneGraph(1, images=DATA_PATH, imageDataDir=DATA_PATH + "by-id/", synsetFile=DATA_PATH + "synsets.json")

    create_data_visual_genome()

    print("end test")

    image_data = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "images_data.p"), "rb"))
    qas = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "qas.p"), "rb"))
    region_interest = cPickle.load(open(os.path.join("Data/VisualGenome/pickles", "region_interest.p"), "rb"))
    print('loading pickles')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    # Get back the ResNet50 base part of a ResNet50 network trained on MS-COCO
    # model_resnet50 = ResNet50(weights='imagenet', include_top=False)
    # model_resnet50.summary()

    img_input = Input(shape=(200, 200, 3), name="image_input")

    net = ModelZoo()
    # Without Top
    model_resnet50 = net.resnet50_base(img_input, trainable=True)
    # Add AVG Pooling Layer
    model_resnet50 = AveragePooling2D((7, 7), name='avg_pool')(model_resnet50)
    # Add the fully-connected layers
    model_resnet50 = Flatten(name='flatten')(model_resnet50)
    output_resnet50 = Dense(classes, activation='softmax', name='fc')(model_resnet50)

    # Define the model
    model = Model(input=img_input, output=output_resnet50)

    # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    model.summary()

    # Load pre-trained weights for ResNet50
    try:
        print('loading weights from {}'.format(config.base_net_weights))
        model.load_weights(config.base_net_weights, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))

    optimizer = Adam(1e-6)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy')

    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
                 ModelCheckpoint(config.model_path, monitor='val_loss', save_best_only=True, verbose=0)]

    print('Starting training')
    model.fit_generator(data_gen_train, samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH, nb_epoch=NUM_EPOCHS,
                        validation_data=data_gen_val, nb_val_samples=NUM_VAL_SAMPLES, callbacks=callbacks,
                        max_q_size=10, nb_worker=1)

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
