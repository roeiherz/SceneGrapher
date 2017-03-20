import cv2
import numpy
import tensorflow as tf
from keras import backend as K
from keras.engine import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, \
    BatchNormalization, \
    Activation, Merge, merge, AveragePooling2D, TimeDistributed
from keras.optimizers import SGD
from DesignPatterns.Singleton import Singleton
from keras_frcnn.Layers.FixedBatchNormalization import FixedBatchNormalization
from keras_frcnn.Layers.RoiPoolingConv import RoiPoolingConv

CHANNEL_AXIS = 3


class ModelZoo(object):
    __metaclass__ = Singleton

    # Class Field
    POOLING_REGIONS = 7

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

        _model.add(ZeroPadding2D((1, 1), input_shape=(None, None, 3)))
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

    def resnet50_classifier(self, base_layers, input_rois, num_rois, num_classes=21, trainable=False):
        """
        This function creates the classifier
        :param base_layers: base layers (conv layers one to five)
        :param input_rois: rois as a keras Input
        :param num_rois: number of rois which defines in the config
        :param num_classes: number of classes
        :param trainable: "freeze" a layer - exclude it from training
        :return: classifier
        """

        def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
            """
            This function creates resnet block with identity shortcut (no conv)
            This blocks sums between   conv 1x1 -> BN -> ReLU ->
                           conv 3x3 -> BN -> ReLU ->
                           conv 1x1 -> BN -> ReLU ->
            and shortcut is identity (f(x)=x)
            :param input_tensor: input tensor
            :param kernel_size: defualt 3, the kernel size of middle conv layer at main path
            :param filters: list of integers, the nb_filters of 3 conv layer at main path
            :param stage: integer, current stage label, used for generating layer names
            :param block: 'a','b'..., current block label, used for generating layer names
            :param trainable:
            :return:
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_dim_ordering() == 'tf':
                bn_axis = 3
            else:
                bn_axis = 1

            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
                                name=conv_name_base + '2a')(input_tensor)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = TimeDistributed(
                Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                              padding='same'), name=conv_name_base + '2b')(x)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                                name=conv_name_base + '2c')(x)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2c')(x)

            x = merge([x, input_tensor], mode='sum')
            x = Activation('relu')(x)

            return x

        def conv_block_td(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
            """
            This function creates a TimeDistributed resnet block with conv as shortcut
            This blocks sums between   conv 1x1 -> BN -> ReLU ->
                                       conv 3x3 -> BN -> ReLU ->
                                       conv 1x1 -> BN -> ReLU ->
                        and shortcut conv 1x1 -> BN
            :param input_tensor: input tensor
            :param kernel_size: kernel size. default is 3 which is the middle conv in the block
            :param filters: list of integers which contains the nof_filters of 3 conv layer
            :param stage: integer, current stage label, used for generating layer names
            :param block: 'a','b'..., current block label, used for generating layer names
            :param strides: stride
            :param trainable: "freeze" a layer - exclude it from training
            # Note that from stage 3, the first conv layer at main path is with strides=(2,2)
            # And the shortcut should have strides=(2,2) as well
            :return:
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_dim_ordering() == 'tf':
                bn_axis = 3
            else:
                bn_axis = 1

            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = TimeDistributed(
                Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                name=conv_name_base + '2a')(input_tensor)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = TimeDistributed(
                Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                              kernel_initializer='normal'), name=conv_name_base + '2b')(x)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'),
                                name=conv_name_base + '2c', trainable=trainable)(x)
            x = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '2c')(x)

            shortcut = TimeDistributed(
                Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
                name=conv_name_base + '1')(input_tensor)
            shortcut = TimeDistributed(FixedBatchNormalization(trainable=False, axis=bn_axis), name=bn_name_base + '1')(
                shortcut)

            x = merge([x, shortcut], mode='sum')
            x = Activation('relu')(x)
            return x

        def classifier_layers(base, trainable=False):
            """

            :param base:
            :param trainable:
            :return:
            """

            x = conv_block_td(base, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), trainable=trainable)
            x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
            x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
            x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

            return x

        out_roi_pool = RoiPoolingConv(self.POOLING_REGIONS, num_rois)([base_layers, input_rois])
        out = classifier_layers(out_roi_pool, trainable=trainable)
        out = TimeDistributed(Flatten(), name='td_flatten')(out)
        out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer='zero'),
                                    name='dense_class_{}'.format(num_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer='zero'),
                                   name='dense_regress_{}'.format(num_classes))(out)

        return [out_class, out_regr]

    def resnet50_base(self, input_tensor, trainable=True):
        """
        This function defines resnet50 base+rpn+classifier as in faster-rcnn
        :param num_rois: number of ROIs
        :param num_anchors: number of anchors: anchor_box_scale * anchor_box_ratio
        :param roi_input: rois as a keras Input
        :param img_input: image Input used to instantiate a keras tensor
        :param trainable: "freeze" a layer - exclude it from training
        :return: full model
        """

        def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
            """
            This function creates resnet block with identity shortcut (no conv)
            This blocks sums between   conv 1x1 -> BN -> ReLU ->
                           conv 3x3 -> BN -> ReLU ->
                           conv 1x1 -> BN -> ReLU ->
            and shortcut is identity (f(x)=x)
            :param input_tensor: input tensor
            :param kernel_size: defualt 3, the kernel size of middle conv layer at main path
            :param filters: list of integers, the nb_filters of 3 conv layer at main path
            :param stage: integer, current stage label, used for generating layer names
            :param block: 'a','b'..., current block label, used for generating layer names
            :param trainable:
            :return:
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_dim_ordering() == 'tf':
                bn_axis = 3
            else:
                bn_axis = 1
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same', name=conv_name_base + '2b',
                              trainable=trainable)(x)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', trainable=trainable)(x)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2c')(x)

            x = merge([x, input_tensor], mode='sum')
            x = Activation('relu')(x)
            return x

        def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
            """
            This function creates resnet block with conv as shortcut
            This blocks sums between   conv 1x1 -> BN -> ReLU ->
                                       conv 3x3 -> BN -> ReLU ->
                                       conv 1x1 -> BN -> ReLU ->
                        and shortcut conv 1x1 -> BN
            :param input_tensor: input tensor
            :param kernel_size: kernel size. default is 3 which is the middle conv in the block
            :param filters: list of integers which contains the nof_filters of 3 conv layer
            :param stage: integer, current stage label, used for generating layer names
            :param block: 'a','b'..., current block label, used for generating layer names
            :param strides: stride
            :param trainable: "freeze" a layer - exclude it from training
            # Note that from stage 3, the first conv layer at main path is with strides=(2,2)
            # And the shortcut should have strides=(2,2) as well
            :return:
            """

            nb_filter1, nb_filter2, nb_filter3 = filters
            if K.image_dim_ordering() == 'tf':
                bn_axis = 3
            else:
                bn_axis = 1

            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'

            x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(
                input_tensor)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2a')(x)
            x = Activation('relu')(x)

            x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                              trainable=trainable)(x)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2b')(x)
            x = Activation('relu')(x)

            x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
            x = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '2c')(x)

            shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1',
                                     trainable=trainable)(input_tensor)
            shortcut = FixedBatchNormalization(trainable=False, axis=bn_axis, name=bn_name_base + '1')(shortcut)

            x = merge([x, shortcut], mode='sum')
            x = Activation('relu')(x)
            return x

        # Determine proper input shape
        if K.image_dim_ordering() == 'th':
            input_shape = (3, None, None)
        else:
            input_shape = (None, None, 3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1

        x = ZeroPadding2D((3, 3))(img_input)

        x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
        x = FixedBatchNormalization(trainable=False, axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

        return x

    def rpn(self, base_layers, num_anchors):
        """
        This function creates the region proposal network
        :param base_layers: the base layer (conv layers one to five)
        :param num_anchors: number of anchors: anchor_box_scale * anchor_box_ratio
        :return: classification and regression layers
        """

        x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal',
                          name='rpn_conv1')(base_layers)
        x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                name='rpn_out_class')(x)
        x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='normal',
                               name='rpn_out_regress')(x)

        return [x_class, x_regr]

        # # define the base network (resnet here, can be VGG, Inception, etc)
        # base_layers = base(img_input, trainable=True)
        # # define the RPN, built on the base layers
        # rpn_layers = rpn(base_layers, num_anchors)
        # # the classifier is build on top of the base layers + the ROI pooling layer + extra layers
        # classifier = classifier(base_layers, roi_input, num_rois, num_classes=nb_classes, trainable=trainable)
        # # define the full model
        # model = Model([img_input, roi_input], rpn_layers + classifier)
        # return model

    def resnet_faster_rcnn(self, img_input, roi_input, num_anchors, num_rois, nb_classes, trainable=True):
        """
        This function defines resnet50 base+rpn+classifier as in faster-rcnn
        :param num_rois: number of ROIs
        :param num_anchors: number of anchors: anchor_box_scale * anchor_box_ratio
        :param roi_input: rois as a keras Input
        :param img_input: image Input used to instantiate a Keras tensor
        :param trainable: "freeze" a layer - exclude it from training
        :return: full model
        """
        # define the base network (resnet here, can be VGG, Inception, etc)
        base_layers = self.resnet50_base(img_input, trainable=trainable)
        # define the RPN, built on the base layers
        rpn_layers = self.rpn(base_layers, num_anchors)
        # the classifier is build on top of the base layers + the ROI pooling layer + extra layers
        classifier = self.resnet50_classifier(base_layers, roi_input, num_rois, num_classes=nb_classes,
                                              trainable=trainable)
        # define the full model
        model = Model([img_input, roi_input], rpn_layers + classifier)
        return model


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(numpy.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = numpy.expand_dims(im, axis=0)
    my_net = ModelZoo()
    # Test pretrained model
    model = my_net.vgg16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print numpy.argmax(out)
