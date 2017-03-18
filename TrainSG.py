from __future__ import print_function

from keras_frcnn.Data.DataGenerator import DataGenerator

__author__ = 'roeih'

import random
import pprint
import os
import sys
import cPickle
import json
from keras_frcnn.Config import Config
# from keras_frcnn import resnet as nn
from keras.optimizers import Adam, SGD
from keras.layers import Input
from keras_frcnn.Data.PascalVoc import PascalVoc
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras_frcnn import losses
# from keras_frcnn import data_generators
from keras import backend as K

SAVE_PATH = "/home/roeih/SceneGrapher/keras_frcnn/Pickle"


def create_data(load=False):
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


if __name__ == '__main__':

    # Load class config
    config = Config()

    train_imgs, val_imgs, hierarchy_mapping, classes_count = create_data(load=True)
    data_gen_train = DataGenerator(data=train_imgs, hierarchy_mapping=hierarchy_mapping, classes_count=classes_count,
                                   config=config, backend=K.image_dim_ordering(), mode='train', batch_size=1)

    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(config.num_rois, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    # the classifier is build on top of the base layers + the ROI pooling layer + extra layers
    classifier = nn.classifier(shared_layers, roi_input, config.num_rois, nb_classes=len(classes_count), trainable=True)

    # define the full model
    model = Model([img_input, roi_input], rpn + classifier)

    try:
        print('loading weights from {}'.format(config.base_net_weights))
        model.load_weights(config.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))

    optimizer = Adam(1e-6)
    model.compile(optimizer=optimizer,
                  loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls,
                        losses.class_loss_regr(config.num_rois, len(classes_count) - 1)])

    nb_epochs = 50

    callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
                 ModelCheckpoint(config.model_path, monitor='val_loss', save_best_only=True, verbose=0)]
    train_samples_per_epoch = 2000  # len(train_imgs)
    nb_val_samples = 500  # len(val_imgs),

    print('Starting training')

    model.fit_generator(data_gen_train, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epochs,
                        validation_data=data_gen_val, nb_val_samples=nb_val_samples, callbacks=callbacks, max_q_size=10,
                        nb_worker=1)
