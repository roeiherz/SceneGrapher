import cv2
import numpy
import tensorflow as tf

from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, BatchNormalization, \
    Activation, Merge
from keras.optimizers import SGD

from DesignPatterns.Singleton import Singleton

CHANNEL_AXIS = 3


class ModelZoo(object):

    __metaclass__ = Singleton

    def billinear(self, output_size):
        _model = Sequential()
        _channel_a = self.vgg19(convolution_only=True)
        _channel_b = self.vgg19(convolution_only=True)

        def _merge_layers(tensor_list):
            if len(tensor_list) != 2:
                raise Exception('Number of input for billiear merge should be 2')

            _arg_a = tf.expand_dims(tf.transpose(tensor_list[0], [0, 2, 3, 1]), 3)
            _arg_b = tf.expand_dims(tf.transpose(tensor_list[1], [0, 2, 3, 1]), 2)

            return tf.multiply(_arg_a, _arg_b)

        _model.add(Merge([_channel_a, _channel_b], _merge_layers))
        _model.add(Lambda(lambda x: x.sum(1).sum(1)))
        _model.add(Lambda(lambda x: tf.sign(x) * tf.sqrt(tf.abs(x))))
        _model.add(Flatten())

        return _model

    def vgg19(self, convolution_only=False):
        _model = Sequential()

        _model.add(ZeroPadding2D((1, 1), input_shape=(None,None,3)))
        _model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        if not convolution_only:
            _model.add(Flatten())
            _model.add(Dense(4096, activation='relu', name='dense_1'))
            _model.add(Dropout(0.5))
            _model.add(Dense(4096, activation='relu', name='dense_2'))
            _model.add(Dropout(0.5))

        return _model

    def vgg16(self, convolution_only=False):

        _model = Sequential()

        _model.add(ZeroPadding2D((1, 1), input_shape=(None, None, 3)))
        _model.add(Convolution2D(64, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(64, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(128, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(128, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(256, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(ZeroPadding2D((1, 1)))
        _model.add(Convolution2D(512, 3, 3))
        _model.add(BatchNormalization(axis=1))
        _model.add(Activation("relu"))
        _model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        if not convolution_only:
            _model.add(Flatten())
            _model.add(Dense(4096))
            _model.add(BatchNormalization())
            _model.add(Activation("relu"))
            _model.add(Dropout(0.5))
            _model.add(Dense(4096))
            _model.add(BatchNormalization())
            _model.add(Activation("relu"))
            _model.add(Dropout(0.5))

        return _model

    def tiny(cls, input_shape, convolution_only=False):

        _model = Sequential()
        _model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
        _model.add(Convolution2D(32, 9, 9, subsample=(3, 3), activation='relu'))
        _model.add(BatchNormalization())
        _model.add(MaxPooling2D((5, 5), strides=(1, 1)))
        _model.add(Convolution2D(32, 1, 1, activation='relu', border_mode='same'))
        _model.add(BatchNormalization())

        if not convolution_only:
            _model.add(Flatten())
            _model.add(Dense(512, activation='relu'))
            _model.add(BatchNormalization())
            _model.add(Dropout(0.5))

        return _model

    def get_model(self, model_name):
        _model_method = getattr(self, model_name)
        return _model_method()

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(numpy.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = numpy.expand_dims(im, axis=0)
    my_net = ModelZoo()
    # Test pretrained model
    model = my_net.vgg16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print numpy.argmax(out)