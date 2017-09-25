import os
import random

import cv2
import numpy as np

from DesignPatterns.Detections import Detections
from FeaturesExtraction.Lib.DataAugmentation import augment_visual_genome
from FeaturesExtraction.Utils.Boxes import iou, BOX, find_union_box
from FeaturesExtraction.Utils.Utils import VG_DATA_PATH, get_mask_from_object, get_img_resize, get_img
from Utils.Logger import Logger

__author__ = 'roeih'


def visual_genome_data_parallel_generator_with_batch(data, hierarchy_mapping, config, mode, batch_size=128):
    """
    This function is a generator for subject and object together with Detections
    :param batch_size: batch size
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    # Flag for final step
    # flag=True
    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:

                imgs = []
                labels = []

                # Check The last step and make sure we are not doing additional step
                # if not flag:
                #     yield [np.copy(imgs)], [np.copy(labels)]

                Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):

                    # the index
                    ind = batch_num * batch_size + current_index

                    # detection per index
                    detection = data[ind]
                    # Get image
                    img = get_img(detection[Detections.Url])

                    if img is None:
                        Logger().log("Coulden't get the image")
                        continue

                    # In-case we want to normalize
                    if config.normalize:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # For-each pairwise objects: once Subject and once Object
                    for i in range(2):

                        if i == 0:
                            # Subject
                            classification = Detections.SubjectClassifications
                            box = Detections.SubjectBox
                        else:
                            # Object
                            classification = Detections.ObjectClassifications
                            box = Detections.ObjectBox

                        # Get the label of object
                        label = detection[classification]

                        # Check if it is a correct label
                        if label not in correct_labels:
                            continue

                        # Get the label uuid
                        label_id = hierarchy_mapping[label]

                        # Create the y labels as a one hot vector
                        y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                        # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                        box = detection[box]

                        # Cropping the patch from the image.
                        patch = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                        # Resize the image according the padding method
                        resized_img = get_img_resize(patch, config.crop_width, config.crop_height,
                                                     type=config.padding_method)

                        if mode == 'train' and config.use_jitter:
                            # Augment only in training
                            # todo: create a regular jitter for each patch increase the number of patches by some constant
                            # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                            Logger().log("No data augmentation")

                        # Expand dimensions - add batch dimension for the numpy
                        resized_img = np.expand_dims(resized_img, axis=0)
                        y_labels = np.expand_dims(y_labels, axis=0)

                        imgs.append(np.copy(resized_img))
                        labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log(
                    "Exception for detection_id: {0}, image: {1}, current batch: {2}".format(detection[Detections.Id],
                                                                                             detection[
                                                                                                 Detections.Url],
                                                                                             batch_num))
                Logger().log(str(e))

                # Check if it is the last batch
                # if batch_num + 1 == num_of_batches_per_epoch:
                #     flag = False


def visual_genome_data_parallel_generator(data, hierarchy_mapping, config, mode):
    """
    This function is a generator for subject and object together with Detections
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test'
    """

    correct_labels = hierarchy_mapping.keys()
    # For printing
    ind = 1
    size = len(data)

    while True:
        for detection in data:
            try:
                Logger().log("Prediction Sample is {0}/{1}".format(ind, size))

                img = get_img(detection[Detections.Url])

                if img is None:
                    Logger().log("Coulden't get the image")
                    continue

                # In-case we want to normalize
                if config.normalize:
                    # Subtract mean and normalize
                    mean_image = np.mean(img, axis=0)
                    img -= mean_image
                    img /= 128.

                    # Zero-center by mean pixel
                    # norm_img = img.astype(np.float32)
                    # norm_img[:, :, 0] -= 103.939
                    # norm_img[:, :, 1] -= 116.779
                    # norm_img[:, :, 2] -= 123.68

                # For-each pairwise objects: once Subject and once Object
                for i in range(2):

                    if i == 0:
                        # Subject
                        classification = Detections.SubjectClassifications
                        box = Detections.SubjectBox
                    else:
                        # Object
                        classification = Detections.ObjectClassifications
                        box = Detections.ObjectBox

                    # Get the label of object
                    label = detection[classification]

                    # Check if it is a correct label
                    if label not in correct_labels:
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = detection[box]

                    # Cropping the patch from the image.
                    patch = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch, config.crop_width, config.crop_height,
                                                 type=config.padding_method)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    yield [np.copy(resized_img)], [np.copy(y_labels)]

                ind += 1
            except Exception as e:
                Logger().log("Exception for image {0}".format(detection.url))
                Logger().log(str(e))


def visual_genome_data_predicate_pairs_generator_with_batch(data, relations_dict, hierarchy_mapping, config, mode,
                                                            batch_size=128, evaluate=False):
    """
    This function is a generator for Predicate  with pair of objects with batch-size
    :param evaluate:  A flag which indicates if we evaluate in PredictFeaturesModule
    :param relations_dict: This dict contains key as pairs - (subject, object) and their values are predicates used for labels
    :param batch_size: batch size
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    current_index = 0

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    # Get detection
                    ind = batch_num * batch_size + current_index
                    detection = data[ind]
                    subject = detection[0]
                    object = detection[1]

                    # Get image - the pair is from the same image
                    img = get_img(subject.url, download=True)

                    if img is None:
                        Logger().log("Coulden't get the image")
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # Get the label of object
                    if (subject.id, object.id) in relations_dict:
                        label = relations_dict[(subject.id, object.id)]

                    else:
                        # Negative label
                        label = "neg"

                    # Check if it is a correct label
                    if label not in correct_labels:
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Calc Union-Box
                    # Get the Subject mask: a dict with {x1,x2,y1,y2}
                    mask_subject = get_mask_from_object(subject)
                    # Saves as a box
                    subject_box = np.array(
                        [mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])

                    # Get the Object mask: a dict with {x1,x2,y1,y2}
                    mask_object = get_mask_from_object(object)
                    # Saves as a box
                    object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])

                    # Get the UNION box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = find_union_box(subject_box, object_box)

                    # Cropping the patch from the image.
                    patch_union = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch_union, config.crop_width, config.crop_height,
                                                 type=config.padding_method)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    detection, batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_predicate_mask_pairs_generator_with_batch(data, relations_dict, hierarchy_mapping, config, mode,
                                                            batch_size=128, evaluate=False):
    """
    This function is a generator for Predicate  with pair of objects with batch-size
    :param evaluate:  A flag which indicates if we evaluate in PredictFeaturesModule
    :param relations_dict: This dict contains key as pairs - (subject, object) and their values are predicates used for labels
    :param batch_size: batch size
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    current_index = 0

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    # Get detection
                    ind = batch_num * batch_size + current_index
                    detection = data[ind]
                    subject = detection[0]
                    object = detection[1]

                    # Get image - the pair is from the same image
                    img = get_img(subject.url, download=True)

                    if img is None:
                        Logger().log("Coulden't get the image")
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # Get the label of object
                    if (subject.id, object.id) in relations_dict:
                        label = relations_dict[(subject.id, object.id)]

                    else:
                        # Negative label
                        label = "neg"

                    # Check if it is a correct label
                    if label not in correct_labels:
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Calc Union-Box
                    # Get the Subject mask: a dict with {x1,x2,y1,y2}
                    mask_subject = get_mask_from_object(subject)
                    # Saves as a box
                    subject_box = np.array(
                        [mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])

                    # Get the Object mask: a dict with {x1,x2,y1,y2}
                    mask_object = get_mask_from_object(object)
                    # Saves as a box
                    object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])

                    # Fill HeatMap
                    heat_map = np.zeros(img.shape)
                    heat_map[subject_box[BOX.Y1]: subject_box[BOX.Y2], subject_box[BOX.X1]: subject_box[BOX.X2],
                    :] = 255
                    heat_map[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2], :] = 255

                    # Get the UNION box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = find_union_box(subject_box, object_box)

                    # Cropping the patch from the image.
                    patch_union = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
                    patch_heatmap = heat_map[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch_union, config.crop_width, config.crop_height,
                                                 type=config.padding_method)
                    resized_heatmap = get_img_resize(patch_heatmap, config.crop_width, config.crop_height,
                                                    type=config.padding_method)

                    # Concatenate the heat-map to the image in the kernel axis
                    resized_img = np.concatenate((resized_img, resized_heatmap[:, :, :1]), axis=2)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    detection, batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_predicate_generator_with_batch(data, hierarchy_mapping, config, mode, classification, type_box,
                                                      batch_size=128, evaluate=False):
    """
    This function is a generator for Predicate with Detections with batch-size
    :param evaluate: A flag which indicates if we evaluate in PredictVisualModule 
    :param batch_size: batch size
    :param type_box: Detections.SubjectBox ('subject_box') or Detections.ObjectBox ('object_box') or
                     Detection.UnionBox ('union_box')
    :param classification: Detections.SubjectClassifications ('subject_classifications') or
            Detections.ObjectClassifications ('object_classifications') or Detections.Predicate ('predicate')
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    # Get detection
                    ind = batch_num * batch_size + current_index
                    detection = data[ind]

                    # Get image
                    img = get_img(detection[Detections.Url], download=True)

                    if img is None:
                        Logger().log("Coulden't get the image in url {}".format(detection[Detections.Url]))
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # Get the label of object
                    label = detection[classification]

                    # Check if it is a correct label
                    if label not in correct_labels:
                        Logger().log("WARNING: label isn't familiar")
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = detection[type_box]

                    # Cropping the patch from the image.
                    patch_subject = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch_subject, config.crop_width, config.crop_height,
                                                 type=config.padding_method)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    detection[Detections.Url], batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_predicate_mask_generator_with_batch(data, hierarchy_mapping, config, mode, classification,
                                                           type_box,
                                                           batch_size=128, evaluate=False):
    """
    This function is a generator for Predicate with Detections with batch-size
    :param evaluate: A flag which indicates if we evaluate in PredictVisualModule
    :param batch_size: batch size
    :param type_box: Detections.SubjectBox ('subject_box') or Detections.ObjectBox ('object_box') or
                     Detection.UnionBox ('union_box')
    :param classification: Detections.SubjectClassifications ('subject_classifications') or
            Detections.ObjectClassifications ('object_classifications') or Detections.Predicate ('predicate')
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    # Get detection
                    ind = batch_num * batch_size + current_index
                    detection = data[ind]

                    # Get image
                    img = get_img(detection[Detections.Url], download=True)

                    if img is None:
                        Logger().log("Coulden't get the image in url {}".format(detection[Detections.Url]))
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # Get the label of object
                    label = detection[classification]

                    # Check if it is a correct label
                    if label not in correct_labels:
                        Logger().log("WARNING: label isn't familiar")
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Get Subject and Object boxes
                    subject_box = detection[Detections.SubjectBox]
                    object_box = detection[Detections.ObjectBox]

                    # Fill HeatMap
                    heat_map = np.zeros(img.shape)
                    heat_map[subject_box[BOX.Y1]: subject_box[BOX.Y2], subject_box[BOX.X1]: subject_box[BOX.X2],
                    :] = 255
                    heat_map[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2], :] = 255

                    # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = detection[type_box]

                    # Cropping the patch from the image.
                    patch_predicate = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
                    patch_heatmap = heat_map[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch_predicate, config.crop_width, config.crop_height,
                                                 type=config.padding_method)
                    resized_heatmap = get_img_resize(patch_heatmap, config.crop_width, config.crop_height,
                                                     type=config.padding_method)

                    # Concatenate the heat-map to the image in the kernel axis
                    resized_img = np.concatenate((resized_img, resized_heatmap[:, :, :1]), axis=2)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    detection[Detections.Url], batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_predicate_mask_dual_generator_with_batch(data, hierarchy_mapping, config, mode, classification,
                                                               type_box, batch_size=128, evaluate=False):
    """
    This function is a generator for Predicate with Detections with batch-size
    :param evaluate: A flag which indicates if we evaluate in PredictVisualModule
    :param batch_size: batch size
    :param type_box: Detections.SubjectBox ('subject_box') or Detections.ObjectBox ('object_box') or
                     Detection.UnionBox ('union_box')
    :param classification: Detections.SubjectClassifications ('subject_classifications') or
            Detections.ObjectClassifications ('object_classifications') or Detections.Predicate ('predicate')
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    # Get detection
                    ind = batch_num * batch_size + current_index
                    detection = data[ind]

                    # Get image
                    img = get_img(detection[Detections.Url], download=True)

                    if img is None:
                        Logger().log("Coulden't get the image in url {}".format(detection[Detections.Url]))
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        # Subtract mean and normalize
                        mean_image = np.mean(img, axis=0)
                        img -= mean_image
                        img /= 128.

                        # Zero-center by mean pixel
                        # norm_img = img.astype(np.float32)
                        # norm_img[:, :, 0] -= 103.939
                        # norm_img[:, :, 1] -= 116.779
                        # norm_img[:, :, 2] -= 123.68

                    # Get the label of object
                    label = detection[classification]

                    # Check if it is a correct label
                    if label not in correct_labels:
                        Logger().log("WARNING: label isn't familiar")
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                    # Get Subject and Object boxes
                    subject_box = detection[Detections.SubjectBox]
                    object_box = detection[Detections.ObjectBox]

                    # Fill HeatMap
                    heat_map_subject = np.zeros(img.shape)
                    heat_map_subject[subject_box[BOX.Y1]: subject_box[BOX.Y2], subject_box[BOX.X1]: subject_box[BOX.X2],
                    :] = 255
                    heat_map_object = np.zeros(img.shape)
                    heat_map_object[object_box[BOX.Y1]: object_box[BOX.Y2], object_box[BOX.X1]: object_box[BOX.X2], :] = 255

                    # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                    box = detection[type_box]

                    # Cropping the patch from the image.
                    patch_predicate = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
                    patch_heatmap_heat_map_subject = heat_map_subject[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]
                    patch_heatmap_heat_map_object = heat_map_object[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch_predicate, config.crop_width, config.crop_height,
                                                 type=config.padding_method)
                    resized_heatmap_subject = get_img_resize(patch_heatmap_heat_map_subject, config.crop_width,
                                                             config.crop_height, type=config.padding_method)
                    resized_heatmap_object = get_img_resize(patch_heatmap_heat_map_object, config.crop_width,
                                                            config.crop_height, type=config.padding_method)

                    # Concatenate the heat-map to the image in the kernel axis
                    resized_img = np.concatenate((resized_img, resized_heatmap_subject[:, :, :1]), axis=2)
                    resized_img = np.concatenate((resized_img, resized_heatmap_object[:, :, :1]), axis=2)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                        Logger().log("No data augmentation")

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    detection[Detections.Url], batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_generator(data, hierarchy_mapping, config, mode, classification, type_box):
    """
    This function is a generator for Detections
    :param type_box: Detections.SubjectBox ('subject_box') or Detections.ObjectBox ('object_box') or
                     Detection.UnionBox ('union_box')
    :param classification: Detections.SubjectClassifications ('subject_classifications') or
            Detections.ObjectClassifications ('object_classifications') or Detections.Predicate ('predicate')
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test' or 'validate'
    """

    correct_labels = hierarchy_mapping.keys()

    while True:
        for detection in data:
            try:

                img = get_img(detection[Detections.Url])

                if img is None:
                    Logger().log("Coulden't get the image")
                    continue

                # In-case we want to normalize
                if config.use_jitter:
                    # Subtract mean and normalize
                    mean_image = np.mean(img, axis=0)
                    img -= mean_image
                    img /= 128.

                    # Zero-center by mean pixel
                    # norm_img = img.astype(np.float32)
                    # norm_img[:, :, 0] -= 103.939
                    # norm_img[:, :, 1] -= 116.779
                    # norm_img[:, :, 2] -= 123.68

                # Get the label of object
                label = detection[classification]

                # Check if it is a correct label
                if label not in correct_labels:
                    continue

                # Get the label uuid
                label_id = hierarchy_mapping[label]

                # Create the y labels as a one hot vector
                y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]

                # Get the box: a BOX (numpy array) with [x1,x2,y1,y2]
                box = detection[type_box]

                # Cropping the patch from the image.
                patch_subject = img[box[BOX.Y1]: box[BOX.Y2], box[BOX.X1]: box[BOX.X2], :]

                # Resize the image according the padding method
                resized_img = get_img_resize(patch_subject, config.crop_width, config.crop_height,
                                             type=config.padding_method)

                if mode == 'train' and config.use_jitter:
                    # Augment only in training
                    # todo: create a regular jitter for each patch increase the number of patches by some constant
                    # resized_img = augment_visual_genome(resized_img, detection, config, mask)
                    Logger().log("No data augmentation")

                # Expand dimensions - add batch dimension for the numpy
                resized_img = np.expand_dims(resized_img, axis=0)
                y_labels = np.expand_dims(y_labels, axis=0)

                yield [np.copy(resized_img)], [np.copy(y_labels)]
            except Exception as e:
                Logger().log("Exception for image {0}".format(detection[Detections.Url]))
                Logger().log(str(e))


def visual_genome_data_cnn_generator_with_batch(data, hierarchy_mapping, config, mode, batch_size=128, evaluate=False):
    """
    This function is a generator for only objects for CNN
    :param evaluate: A flag which indicates if we evaluate in PredictFeaturesModule 
    :param batch_size: batch size
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test'
    """

    correct_labels = hierarchy_mapping.keys()
    size = len(data)

    # The number of batches per epoch depends if size % batch_size == 0
    if size % batch_size == 0:
        num_of_batches_per_epoch = size / batch_size
    else:
        num_of_batches_per_epoch = size / batch_size + 1

    while True:

        # Batch number
        for batch_num in range(num_of_batches_per_epoch):
            try:
                imgs = []
                labels = []

                if evaluate:
                    Logger().log("Prediction Batch Number is {0}/{1}".format(batch_num + 1, num_of_batches_per_epoch))

                # Define number of samples per batch
                if batch_size * (batch_num + 1) >= size:
                    nof_samples_per_batch = size - batch_size * batch_num
                else:
                    nof_samples_per_batch = batch_size

                # Start one batch
                for current_index in range(nof_samples_per_batch):
                    ind = batch_num * batch_size + current_index
                    object = data[ind]
                    img = get_img(object.url, download=True)

                    if img is None:
                        Logger().log("Coulden't get the image")
                        continue

                    # In-case we want to normalize
                    if config.use_jitter:
                        img = config.jitter.random_transform(img.astype("float32"))
                        img = config.jitter.standardize(img)

                    # Get the lable of object
                    label = object.names[0]

                    # Check if it is a correct label
                    if label not in correct_labels:
                        continue

                    # Get the label uuid
                    label_id = hierarchy_mapping[label]

                    # Create the y labels as a one hot vector
                    # y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]
                    y_labels = np.zeros((len(hierarchy_mapping)), dtype='uint8')
                    y_labels[label_id] = 1

                    # Get the mask: a dict with {x1,x2,y1,y2}
                    mask = get_mask_from_object(object)

                    # Cropping the patch from the image.
                    patch = img[mask['y1']: mask['y2'], mask['x1']: mask['x2'], :]

                    # Resize the image according the padding method
                    resized_img = get_img_resize(patch, config.crop_width, config.crop_height,
                                                 type=config.padding_method)

                    if mode == 'train' and config.use_jitter:
                        # Augment only in training
                        # todo: create a regular jitter for each patch increase the number of patches by some constant
                        resized_img = augment_visual_genome(resized_img, object, config, mask)

                    # Expand dimensions - add batch dimension for the numpy
                    resized_img = np.expand_dims(resized_img, axis=0)
                    y_labels = np.expand_dims(y_labels, axis=0)

                    imgs.append(np.copy(resized_img))
                    labels.append(np.copy(y_labels))

                # Continue if imgs and labels are empty
                if len(imgs) == 0 or len(labels) == 0:
                    continue

                # Finished one batch
                yield np.concatenate(imgs, axis=0), np.concatenate(labels, axis=0)

            except Exception as e:
                Logger().log("Exception for image {0} in current batch: {1} and number of samples in batch: {2}".format(
                    object.url, batch_num, current_index))
                Logger().log(str(e))


def visual_genome_data_cnn_generator(data, hierarchy_mapping, config, mode):
    """
    This function is a generator for only objects for CNN
    :param data: dictionary of Data
    :param hierarchy_mapping: hierarchy mapping
    :param config: the class config which contains different parameters
    :param mode: 'train' or 'test'
    """

    correct_labels = hierarchy_mapping.keys()

    while True:
        for object in data:
            try:

                img = get_img(object.url)

                if img is None:
                    Logger().log("Coulden't get the image")
                    continue

                # In-case we want to normalize
                if config.use_jitter:
                    # Subtract mean and normalize
                    mean_image = np.mean(img, axis=0)
                    img -= mean_image
                    img /= 128.

                    # Zero-center by mean pixel
                    # norm_img = img.astype(np.float32)
                    # norm_img[:, :, 0] -= 103.939
                    # norm_img[:, :, 1] -= 116.779
                    # norm_img[:, :, 2] -= 123.68

                # Get the lable of object
                label = object.names[0]

                # Check if it is a correct label
                if label not in correct_labels:
                    continue

                # Get the label uuid
                label_id = hierarchy_mapping[label]

                # Create the y labels as a one hot vector
                # y_labels = np.eye(len(hierarchy_mapping), dtype='uint8')[label_id]
                y_labels = np.zeros((len(hierarchy_mapping)), dtype='uint8')
                y_labels[label_id] = 1

                # Get the mask: a dict with {x1,x2,y1,y2}
                mask = get_mask_from_object(object)

                # Cropping the patch from the image.
                patch = img[mask['y1']: mask['y2'], mask['x1']: mask['x2'], :]

                # Resize the image according the padding method
                resized_img = get_img_resize(patch, config.crop_width, config.crop_height,
                                             type=config.padding_method)

                if mode == 'train' and config.use_jitter:
                    # Augment only in training
                    # todo: create a regular jitter for each patch increase the number of patches by some constant



                    resized_img = augment_visual_genome(resized_img, object, config, mask)

                # Expand dimensions - add batch dimension for the numpy
                resized_img = np.expand_dims(resized_img, axis=0)
                y_labels = np.expand_dims(y_labels, axis=0)

                yield [np.copy(resized_img)], [np.copy(y_labels)]
            except Exception as e:
                Logger().log("Exception for image {0}".format(object.url))
                Logger().log(str(e))
