__author__ = 'roeih'

from keras import backend as K


class Config:
    """
    This class represents Config file
    """
    def __init__(self):
        # setting for Data augmentation

        # todo: create a jitter class for future use
        self.jitter = False
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.scale_augment = False
        self.random_rotate = False
        self.random_rotate_scale = 180
        self.dataset = "VisualGenome"

        # anchor box scales
        self.anchor_box_scales = [128, 256, 512]

        # anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        # size to resize the smallest side of the image
        self.im_size = 600

        self.image_width = 800
        self.image_height = 600
        # size to resize
        # todo: need to be decided
        self.crop_width = 224
        self.crop_height = 224
        self.padding_method = "zero_pad"

        # Load weights
        self.load_weights = False

        # number of ROIs at once
        self.num_rois = 2

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = True

        # scaling the stdev
        self.std_scaling = 4.0

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # location of pretrained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        if K.image_dim_ordering() == 'th':
            self.base_net_weights = 'Weights/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
        else:
            self.base_net_weights = 'Weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        self.model_path = 'model_frcnn.hdf5'
