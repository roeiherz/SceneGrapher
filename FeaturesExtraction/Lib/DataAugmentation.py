import numpy as np
import cv2
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis, ImageDataGenerator
from Utils.Logger import Logger

__author__ = 'roeih'

S = 1
LMBDA = 10
EPSILON = 0.000000001


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    """
    Random Erasing is a kind of image augmentation methods for convolutional neural networks (CNN).
    It tries to regularize models using training images that are randomly masked with random values.
    Please refer to this repository https://github.com/yu4u/cutout-random-erasing
    for the details of algorithm and its implementation.
    :param p:
    :param s_l:
    :param s_h:
    :param r_1:
    :param r_2:
    :param v_l:
    :param v_h:
    :return:
    """
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def augment_pascal_voc(img, img_data, config):
    """
    This function augment the data according to the config file
    :param img: an image numpy type
    :param img_data: image data
    :param config: a config class
    :return:
    """
    img_data_aug = np.copy(img_data)
    rows, cols = img.shape[:2]

    if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 1)
        for bbox in img_data_aug['bboxes']:
            x1 = bbox['x1']
            x2 = bbox['x2']
            bbox['x2'] = cols - x1
            bbox['x1'] = cols - x2

    if config.use_vertical_flips and np.random.randint(0, 2) == 0:
        img = cv2.flip(img, 0)
        for bbox in img_data_aug['bboxes']:
            y1 = bbox['y1']
            y2 = bbox['y2']
            bbox['y2'] = rows - y1
            bbox['y1'] = rows - y2

    if config.random_rotate:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                    np.random.randint(-config.random_rotate_scale,
                                                      config.random_rotate_scale), 1)
        img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

        for bbox in img_data_aug['bboxes']:
            K = np.array([[bbox['x1'], bbox['y1']], [bbox['x2'], bbox['y2']], [bbox['x1'], bbox['y2']],
                          [bbox['x2'], bbox['y1']]])
            K = cv2.transform(K.reshape(4, 1, 2), M)[:, 0, :]

            (x1, y1) = np.min(K, axis=0)
            (x2, y2) = np.max(K, axis=0)

            bbox['x1'] = x1
            bbox['x2'] = x2
            bbox['y1'] = y1
            bbox['y2'] = y2

    return img_data_aug, img


def augment_visual_genome(patch, object, config, mask):
    """
    This function augment the data according to the config file
    :param patch: an image numpy type
    :param object: object Visual Genome class
    :param config: a config class
    :param mask: a dict with {x1,x2,y1,y2} for each object
    :return:
    """
    rows, cols = patch.shape[:2]
    new_patch = patch

    if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
        new_patch = cv2.flip(patch, 1)

    if config.use_vertical_flips and np.random.randint(0, 2) == 0:
        new_patch = cv2.flip(patch, 0)

    if config.random_rotate:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                    np.random.randint(-config.random_rotate_scale,
                                                      config.random_rotate_scale), 1)
        new_patch = cv2.warpAffine(patch, M, (cols, rows), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

        # new_patches.append(new_patch)
        # new_labels.append(object.names[0])

        # K = np.array([[mask['x1'], mask['y1']], [mask['x2'], mask['y2']], [mask['x1'], mask['y2']],
        #               [mask['x2'], mask['y1']]])
        # K = cv2.transform(K.reshape(4, 1, 2), M)[:, 0, :]

        # (x1, y1) = np.min(K, axis=0)
        # (x2, y2) = np.max(K, axis=0)
        #
        # mask['x1'] = x1
        # mask['x2'] = x2
        # mask['y1'] = y1
        # mask['y2'] = y2

    return new_patch


def fix_orientation(image, degree):
    if degree == 0:
        return image
    else:
        return np.rot90(image, degree / 90)


class Jitter(object):
    """
    This class is represents the different data augmentation that one can do in image
    """

    def __init__(self, crop_width, crop_height, padding_method):
        self.keras_jitter = None
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.padding_method = padding_method
        self.use_keras_jitter = False
        self.use_mixup = False

    def set_keras_jitter(self, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False,
                         preprocessing_function=None):
        """
        This function set the Keras Jitter: ImageDataGenerator
        :param width_shift_range: fraction of total width.
        :param height_shift_range: fraction of total height.
        :param horizontal_flip: whether to randomly flip images horizontally.
        :param preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        :return:
        """
        if self.keras_jitter is None:
            self.keras_jitter = ImageDataGenerator(width_shift_range=width_shift_range,
                                                   height_shift_range=height_shift_range,
                                                   horizontal_flip=horizontal_flip,
                                                   preprocessing_function=preprocessing_function)
            # Change the flag
            self.use_keras_jitter = True
        else:
            Logger().log("Error: Keras Jitter is already have been set.")

    def set_mixup_jitter(self, mixup_alpha=0.1):
        """
        This function set the MixUp Jitter.
        :return:
        """
        self.use_mixup = True
        self._mixup_alpha = mixup_alpha

    def mixup(self, batchsize, x1, x2):
        """
        This function implements H Zhang, M Cisse, YN Dauphin and D Lopez-Paz (2017) mixup: Beyond Empirical Risk Minimization
        :return:
        """
        lam = np.random.beta(self._mixup_alpha + 1, self._mixup_alpha)

        x = x1 * lam + x2 * (1 - lam)
        return x

    def apply_jitter(self, resized_img, batchsize, new_resized_img=None):
        """
        This function apply Jitter depends which jitter one want to use
        :param resized_img: the resize img which we want to jitter
        :param batchsize: the size of the batch size
        :param new_resized_img: a new resize img (for mixup Jitter)
        :return:
        """

        # Use mixup augmentation
        # if self.use_mixup:
        #     resized_img = self.mixup(batchsize=batchsize, x1=resized_img, x2=new_resized_img)

        # Use Keras Jitter: Random Transformation and Standardization
        if self.use_keras_jitter:
            resized_img = self.keras_jitter.random_transform(resized_img)
            resized_img = self.keras_jitter.standardize(resized_img)

        return resized_img


# class Jitter(object):
#     """
#     This class is represents the different data augmentation that one can do in image
#
#     # Arguments
#         samplewise_center: set each sample mean to 0.
#         samplewise_std_normalization: divide each input by its std.
#         zca_whitening: apply ZCA whitening.
#         rotation_range: degrees (0 to 180).
#         width_shift_range: fraction of total width.
#         height_shift_range: fraction of total height.
#         fill_mode: points outside the boundaries are filled according to the
#             given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
#             is 'nearest'.
#         cval: value used for points outside the boundaries when fill_mode is
#             'constant'. Default is 0.
#         horizontal_flip: whether to randomly flip images horizontally.
#         vertical_flip: whether to randomly flip images vertically.
#         rescale: rescaling factor. If None or 0, no rescaling is applied,
#             otherwise we multiply the data by the value provided
#             (before applying any other transformation).
#         preprocessing_function: function that will be implied on each input.
#             The function will run before any other modification on it.
#             The function should take one argument:
#             one image (Numpy tensor with rank 3),
#             and should output a Numpy tensor with the same shape.
#     """
#
#     def __init__(self,
#                  gcn=False,
#                  samplewise_center=False,
#                  samplewise_std_normalization=False,
#                  global_contrast_normalization=False,
#                  width_shift_range=0.,
#                  height_shift_range=0.,
#                  zca_whitening=False,
#                  rotation_range=0.,
#                  rescale=None,
#                  horizontal_flip=False,
#                  vertical_flip=False,
#                  preprocessing_function=None,
#                  fill_mode='nearest',
#                  orientation=0.,
#                  cval=0.):
#
#         self.gcn = gcn
#         self.samplewise_center = samplewise_center
#         self.samplewise_std_normalization = samplewise_std_normalization
#         self.global_contrast_normalization = global_contrast_normalization
#         self.zca_whitening = zca_whitening
#         self.width_shift_range = width_shift_range
#         self.height_shift_range = height_shift_range
#         self.horizontal_flip = horizontal_flip
#         self.vertical_flip = vertical_flip
#         self.rescale = rescale
#         self.preprocessing_function = preprocessing_function
#         self.orientation = orientation
#         self.rotation_range = rotation_range
#         self.fill_mode = fill_mode
#         self.cval = cval
#
#         # Channels is last in tensorflow as a backend
#         self.channel_axis = 3
#         self.row_axis = 1
#         self.col_axis = 2
#
#         self.mean = None
#         self.std = None
#         self.principal_components = None
#
#     def random_transform(self, x):
#         """Randomly augment a single image tensor.
#
#         # Arguments
#             x: 3D tensor, single image.
#
#         # Returns
#             A randomly transformed version of the input (same shape).
#         """
#         # x is a single image, so it doesn't have image number at index 0
#         img_row_axis = self.row_axis - 1
#         img_col_axis = self.col_axis - 1
#         img_channel_axis = self.channel_axis - 1
#
#         # use composition of homographies
#         # to generate final transform that needs to be applied
#         if self.rotation_range:
#             theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
#         else:
#             theta = 0
#
#         if self.height_shift_range:
#             tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
#         else:
#             tx = 0
#
#         if self.width_shift_range:
#             ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
#         else:
#             ty = 0
#
#         transform_matrix = None
#         if theta != 0:
#             rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                         [np.sin(theta), np.cos(theta), 0],
#                                         [0, 0, 1]])
#             transform_matrix = rotation_matrix
#
#         if tx != 0 or ty != 0:
#             shift_matrix = np.array([[1, 0, tx],
#                                      [0, 1, ty],
#                                      [0, 0, 1]])
#             transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
#
#         if transform_matrix is not None:
#             h, w = x.shape[img_row_axis], x.shape[img_col_axis]
#             transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
#             x = apply_transform(x, transform_matrix, img_channel_axis,
#                                 fill_mode=self.fill_mode, cval=self.cval)
#
#         if self.horizontal_flip:
#             if np.random.random() < 0.5:
#                 x = flip_axis(x, img_col_axis)
#
#         if self.vertical_flip:
#             if np.random.random() < 0.5:
#                 x = flip_axis(x, img_row_axis)
#
#         return x
#
#     def standardize(self, x):
#         """Apply the normalization configuration to a single image.
#
#         # Arguments
#             x: batch of inputs to be normalized.
#
#         # Returns
#             The inputs, normalized.
#         """
#         if self.preprocessing_function:
#             x = self.preprocessing_function(x)
#         if self.rescale:
#             x *= self.rescale
#         # x is a single image, so it doesn't have image number at index 0
#
#         if self.gcn:
#             img_row_index = self.row_axis - 1
#             img_col_index = self.col_axis - 1
#             img_channel_axis = self.channel_axis - 1
#
#             X_average = np.mean(x)
#             x = x - X_average
#
#             # `su` is here the mean, instead of the sum
#             contrast = np.sqrt(LMBDA + np.mean(x ** 2))
#
#             x = S * x / max(contrast, EPSILON)
#
#         if self.samplewise_center:
#             x -= np.mean(x, axis=(img_channel_axis, img_row_index, img_col_index), keepdims=True)
#         if self.samplewise_std_normalization:
#             x /= (np.std(x, axis=(img_channel_axis, img_row_index, img_col_index), keepdims=True) + 1e-7)
#         #
#         #  if self.featurewise_center:
#         #     if self.mean is not None:
#         #         x -= self.mean
#         #     else:
#         #         Logger().get_logger().log("featurewise_center is ")
#         # if self.featurewise_std_normalization:
#         #     if self.std is not None:
#         #         x /= (self.std + 1e-7)
#         #     else:
#         #         Logger().get_logger().log('This ImageDataGenerator specifies '
#         #                                   '`featurewise_std_normalization`, but it hasn\'t'
#         #                                   'been fit on any training data. Fit it '
#         #                                   'first by calling `.fit(numpy_data)`.')
#
#         if self.zca_whitening:
#             if self.principal_components is not None:
#                 flatx = np.reshape(x, (x.size))
#                 whitex = np.dot(flatx, self.principal_components)
#                 x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
#             else:
#                 Logger().get_logger().log('This ImageDataGenerator specifies '
#                                           '`zca_whitening`, but it hasn\'t'
#                                           'been fit on any training data. Fit it '
#                                           'first by calling `.fit(numpy_data)`.')
#         return x
