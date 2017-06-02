import matplotlib as mpl
mpl.use('Agg')
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.optimizers import Adam

from Data.VisualGenome.models import ObjectMapping, RelationshipMapping
from DesignPatterns.Detections import Detections
from keras_frcnn.Lib.VisualGenomeDataGenerator import visual_genome_data_generator, \
    visual_genome_data_parallel_generator, get_img, visual_genome_data_generator_with_batch
from keras_frcnn.Lib.Zoo import ModelZoo
from keras.applications.resnet50 import ResNet50
import os
import cPickle
import numpy as np
from keras_frcnn.Lib.Config import Config
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Model
import sys
import matplotlib.pyplot as plt

from keras_frcnn.Utils.Boxes import find_union_box, BOX
from keras_frcnn.Utils.Utils import VisualGenome_PICKLES_PATH, VG_VisualModule_PICKLES_PATH, get_mask_from_object, \
    get_img_resize, get_time_and_date, TRAINING_PREDICATE_CNN_PATH, create_folder
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2
from keras_frcnn.Utils.Visualizer import VisualizerDrawer, CvColor
from keras_frcnn.Utils.data import get_sorted_data, generate_new_hierarchy_mapping, splitting_to_datasets, \
    get_predicate_hierarchy_mapping_from_detections, process_to_detections, get_filtered_data

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2
NUM_EPOCHS = 90
NUM_BATCHES = 128

# If the allocation of training, validation and testing does not adds up to one
used_percent = TRAINING_PERCENT + VALIDATION_PERCENT + TESTING_PERCENT
if not used_percent == 1:
    error_msg = 'Data used percent (train + test + validation) is {0} and should be 1'.format(used_percent)
    print(error_msg)
    raise Exception(error_msg)

__author__ = 'roeih'


def preprocessing_objects(img_data, hierarchy_mapping, object_file_name='objects.p'):
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


def preprocessing_relations(img_data, hierarchy_mapping_objects, hierarchy_mapping_predicates,
                            relation_file_name='full_relations.p'):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param relation_file_name: relation pickle file name
    :param img_data: list of entities files
    :param hierarchy_mapping_objects: dict of objects hierarchy_mapping
    :param hierarchy_mapping_predicates: dict of predicates hierarchy_mapping
    :return: list of RelationshipMapping
    """

    # Check if pickles are already created
    objects_path = os.path.join(VG_VisualModule_PICKLES_PATH, relation_file_name)

    if os.path.isfile(objects_path):
        print('File is already exist {0}'.format(objects_path))
        objects = cPickle.load(open(objects_path, 'rb'))
        return objects

    # Get the whole objects from entities
    objects_lst = []
    correct_object_labels = hierarchy_mapping_objects.keys()
    correct_predicates_labels = hierarchy_mapping_predicates.keys()
    idx = 0
    for img in img_data:

        # Get the url image
        url = img.image.url
        # Get the objects per image
        relations = img.relationships
        for relation in relations:

            # Get the label of object1 and object2
            label_o1 = relation.object.names[0]
            label_o2 = relation.subject.names[0]
            label_predicate = relation.predicate

            # Check if it is a correct object label
            if label_o1 not in correct_object_labels or label_o2 not in correct_object_labels:
                continue

            # Check if it is a correct predicate label
            if label_predicate not in correct_predicates_labels:
                continue

            new_relation_mapping = RelationshipMapping(relation.id, relation.subject, relation.predicate,
                                                       relation.object, relation.synset, url, relation.filtered_id)
            # Append the new objectMapping to objects_lst
            objects_lst.append(new_relation_mapping)

        idx += 1
        print("Finished img: {}".format(idx))

    # Save the objects files to the disk
    objects_file = open(objects_path, 'wb')
    # Pickle objects_lst
    objects_array = np.array(objects_lst)
    cPickle.dump(objects_array, objects_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    objects_file.close()
    return objects_array


def get_classes_mapping_and_hierarchy_mapping_by_objects(objects, path):
    """
    This function creates classes_mapping and hierarchy_mapping by objects and updates the hierarchy_mapping accordingly
    :param objects: list of objects
    :param path: saving or loading the classes_count_per_objects and hierarchy_mapping_per_objects from path folder
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

    # Printing which GPU you have selected
    print("Selected GPU number: {0}".format(gpu_num))

    # Load class config
    config = Config(gpu_num)

    # Define GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)

    # Define tensorflow use only the amount of memory required for the process
    # config_tf = tf.ConfigProto()
    # config_tf.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config_tf))

    # Get time and date
    time_and_date = get_time_and_date()
    # Path for the training folder
    path = os.path.join(TRAINING_PREDICATE_CNN_PATH, time_and_date)
    # Create a new folder for training
    create_folder(path)
    # loading model weights
    if config.loading_model:
        net_weights = os.path.join(config.loading_model_folder, config.model_weights_name)
        print("Loading Weights from: {}".format(net_weights))
    else:
        # The Weights for training
        net_weights = config.base_net_weights
        print("Taking Base Weights from: {}".format(net_weights))
    net_weights_path = os.path.join(path, config.model_weights_name)
    print("The new Model Weights will be Saved: {}".format(net_weights_path))

    # classes_count, hierarchy_mapping, entities = get_sorted_data(classes_count_file_name="final_classes_count.p",
    #                                                              hierarchy_mapping_file_name="final_class_mapping.p",
    #                                                              entities_file_name="final_entities.p",
    #                                                              nof_labels=NOF_LABELS)

    # Load filtered data
    entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(filtered_data_file_name=
                                                                                          "filtered_module_data.p")

    # Get Visual Genome Data relations
    relations = preprocessing_relations(entities, hierarchy_mapping_objects, hierarchy_mapping_predicates,
                                        relation_file_name="mini_filtered_relations.p")

    # Process relations to numpy Detections dtype
    detections = process_to_detections(relations, detections_file_name="mini_filtered_detections.p")
    # Split the data to train, test and validate
    train_imgs, test_imgs, val_imgs = splitting_to_datasets(detections, training_percent=TRAINING_PERCENT,
                                                            testing_percent=TESTING_PERCENT, num_epochs=NUM_EPOCHS,
                                                            path=path, config=config)

    # Get the predicate hierarchy mapping and the number of the predicated classes
    # predicate_classes_count, predicate_hierarchy_mapping = get_predicate_hierarchy_mapping_from_detections(detections,
    #                                                                                                        path,
    #                                                                                                        config=config)

    # Create a data generator for VisualGenome without batch num
    # data_gen_train_vg = visual_genome_data_generator(data=train_imgs,
    #                                                  hierarchy_mapping=hierarchy_mapping_predicates,
    #                                                  config=config, mode='train',
    #                                                  classification=Detections.Predicate, type_box=Detections.UnionBox)
    #
    # # Create a data generator for VisualGenome
    # data_gen_test_vg = visual_genome_data_generator(data=test_imgs,
    #                                                 hierarchy_mapping=hierarchy_mapping_predicates,
    #                                                 config=config, mode='test', classification=Detections.Predicate,
    #                                                 type_box=Detections.UnionBox)
    #
    # # Create a data generator for VisualGenome
    # data_gen_validation_vg = visual_genome_data_generator(data=val_imgs,
    #                                                       hierarchy_mapping=hierarchy_mapping_predicates,
    #                                                       config=config, mode='valid',
    #                                                       classification=Detections.Predicate,
    #                                                       type_box=Detections.UnionBox)

    # Create a data generator for VisualGenome with batch size
    data_gen_train_vg = visual_genome_data_generator_with_batch(data=train_imgs,
                                                                hierarchy_mapping=hierarchy_mapping_predicates,
                                                                config=config, mode='train',
                                                                classification=Detections.Predicate,
                                                                type_box=Detections.UnionBox, batch_size=NUM_BATCHES)

    # Create a data generator for VisualGenome
    data_gen_test_vg = visual_genome_data_generator_with_batch(data=test_imgs,
                                                               hierarchy_mapping=hierarchy_mapping_predicates,
                                                               config=config, mode='test',
                                                               classification=Detections.Predicate,
                                                               type_box=Detections.UnionBox, batch_size=NUM_BATCHES)

    # Create a data generator for VisualGenome
    data_gen_validation_vg = visual_genome_data_generator_with_batch(data=val_imgs,
                                                                     hierarchy_mapping=hierarchy_mapping_predicates,
                                                                     config=config, mode='valid',
                                                                     classification=Detections.Predicate,
                                                                     type_box=Detections.UnionBox,
                                                                     batch_size=NUM_BATCHES)

    # Set the number of classes
    number_of_classes = len(hierarchy_mapping_predicates)

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

    # Load pre-trained weights for ResNet50
    try:
        if config.load_weights:
            print('loading weights from {}'.format(net_weights))
            model.load_weights(net_weights, by_name=True)
    except Exception as e:
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        raise Exception(e)

    optimizer = Adam(1e-6)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ModelCheckpoint(net_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir="logs", write_graph=True, write_images=True),
                 CSVLogger(os.path.join(path, 'training.log'), separator=',', append=False)]

    print('Starting training')
    history = model.fit_generator(data_gen_train_vg, steps_per_epoch=len(train_imgs)/NUM_BATCHES, epochs=NUM_EPOCHS,
                                  validation_data=data_gen_test_vg, validation_steps=len(test_imgs)/NUM_BATCHES,
                                  callbacks=callbacks, max_q_size=1, workers=1)

    # Validating the model
    test_score = model.evaluate_generator(data_gen_validation_vg, steps=len(val_imgs)/NUM_BATCHES, max_q_size=1, workers=1)
    # Plot the Score
    print("The Validation loss is: {0} and the Validation Accuracy is: {1}".format(test_score[0], test_score[1]))

    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, "model_accuracy.jpg"))
    plt.close()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(path, "model_loss.jpg"))
    plt.close()
