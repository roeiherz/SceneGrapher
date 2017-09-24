from FeaturesExtraction.Lib.DataAugmentation import Jitter

__author__ = 'roeih'

from keras import backend as K


class Config:
    """
    This class represents Config file
    """

    def __init__(self, gpu_num):

        # Do we continuing to train or start from fresh
        self.loading_model = True
        self.loading_model_folder = "FilesManager/FeaturesExtraction/PredicatesMaskCNN/Sat_Sep_16_14:36:13_2017"
        self.loading_model_token = "scene_graph_base_module.visual_module.predicate_mask_cnn"
        self.model_weights_name = 'model_vg_resnet50.hdf5'
        # Get the cached data-sets and cached hierarchy mapping and class counting
        self.use_cache_dir = False
        # Load weights
        self.load_weights = True
        # Replace the Dense layer
        self.replace_top = False
        # If we replace top, what is the old top number of classes
        if self.replace_top:
            self.nof_classes = 50

        # The Training is only with positive samples
        self.only_pos = False

        # Use all objects data
        self.use_all_objects_data = True

        # # Use masking for predicates CNN network
        # self.use_mask_predicate = True
        #
        # if self.use_mask_predicate:
        #     self.num_of_channels = 4
        # else:
        #     self.num_of_channels = 3

        # location of pre-trained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        if K.image_dim_ordering() == 'th':
            self.base_net_weights = 'scene_graph_base_module.visual_module.image_net_th'
        else:
            self.base_net_weights = "scene_graph_base_module.visual_module.image_net_tf"

        # Use of Jitter
        self.use_jitter = False
        self.use_translation_jitter = False
        self.use_rotation_jitter = False
        self.use_flip_jitter = False
        # Set each sample mean to 0.
        self.use_samplewise_center = False
        # Divide each sample by its std.
        self.use_samplewise_std_normalization = False
        # Global Contrast Normalization
        self.use_gcn =False

        # Set Jitter
        self.Set_Jitter()

        # Define the GPU number
        self.gpu_num = gpu_num

        # For debugging
        self.debug = False

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

    def Set_Jitter(self):
        """
        This function set the jitter
        """
        # Setting for Data augmentation
        if self.use_jitter and self.use_rotation_jitter:
            rotation_range = 90.
        else:
            rotation_range = 0.
        if self.use_jitter and self.use_translation_jitter:
            width_shift_range = 0.2
            height_shift_range = 0.2
        else:
            width_shift_range = 0.
            height_shift_range = 0.
        if self.use_jitter and self.use_samplewise_center:
            samplewise_center = True
        else:
            samplewise_center = False
        if self.use_jitter and self.use_samplewise_std_normalization:
            samplewise_std_normalization = True
        else:
            samplewise_std_normalization = False

        if self.use_gcn:
            gcn = True
        else:
            gcn = False

        self.jitter = Jitter(samplewise_center=samplewise_center,
                             samplewise_std_normalization=samplewise_std_normalization,
                             global_contrast_normalization=False,
                             width_shift_range=width_shift_range,
                             height_shift_range=height_shift_range,
                             zca_whitening=False,
                             rotation_range=rotation_range,
                             rescale=None,
                             horizontal_flip=False,
                             vertical_flip=False,
                             preprocessing_function=None,
                             fill_mode='nearest',
                             cval=0.)
