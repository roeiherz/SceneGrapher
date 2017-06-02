import matplotlib as mpl

mpl.use('Agg')
from Data.VisualGenome.models import ObjectMapping
from keras_frcnn.Lib.PascalVocDataGenerator import PascalVocDataGenerator
from keras_frcnn.Lib.VisualGenomeDataGenerator import visual_genome_data_cnn_generator, \
    visual_genome_data_cnn_generator_with_batch
from keras_frcnn.Lib.Zoo import ModelZoo
from keras.applications.resnet50 import ResNet50
import os
import cPickle
import numpy as np
from keras_frcnn.Lib.Config import Config
from keras.optimizers import Adam
from keras.layers import Input, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras import backend as K
from keras.models import Model
import sys
import matplotlib.pyplot as plt
from keras_frcnn.Utils.Utils import VisualGenome_PICKLES_PATH, get_time_and_date, create_folder, \
    TRAINING_OBJECTS_CNN_PATH, CLASSES_COUNT_FILE, CLASSES_MAPPING_FILE, replace_top_layer
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras_frcnn.Utils.data import get_sorted_data, splitting_to_datasets, create_data_pascal_voc, \
    generate_new_hierarchy_mapping, get_filtered_data

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2
NUM_EPOCHS = 90
NUM_BATCHES = 128
MAX_NOF_SAMPLES_THR = 1000000
MAX_NOF_SAMPLES = 900000

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


def get_classes_mapping_and_hierarchy_mapping_by_objects(objects, path, config=None):
    """
    This function creates classes_mapping and hierarchy_mapping by objects and updates the hierarchy_mapping accordingly
    :param config: config
    :param objects: list of objects
    :param path: saving or loading the classes_count_per_objects and hierarchy_mapping_per_objects from path folder
    :return: dict of classes_mapping and hierarchy_mapping
    """

    # Load hierarchy mapping and class counting from cache
    if config is not None and config.use_cache_dir:
        classes_count_path = os.path.join(config.loading_model_folder, CLASSES_COUNT_FILE)
        hierarchy_mapping_path = os.path.join(config.loading_model_folder, CLASSES_MAPPING_FILE)
        print("Loading from cached hierarchy mapping from {0} and class counting {1}".format(hierarchy_mapping_path,
                                                                                             classes_count_path))
        classes_count_per_objects = cPickle.load(open(classes_count_path, 'rb'))
        hierarchy_mapping_per_objects = cPickle.load(open(hierarchy_mapping_path, 'rb'))
        return classes_count_per_objects, hierarchy_mapping_per_objects

    classes_count_per_objects = {}
    hierarchy_mapping_per_objects = {}
    new_obj_id = 0
    for object in objects:
        # Get the label of object
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

    # Save classes_count_per_objects file
    classes_count_file = file(os.path.join(path, CLASSES_COUNT_FILE), 'wb')
    # Pickle classes_count_per_objects
    cPickle.dump(classes_count_per_objects, classes_count_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    classes_count_file.close()
    # Save hierarchy_mapping_per_objects file
    hierarchy_mapping_file = file(os.path.join(path, CLASSES_MAPPING_FILE), 'wb')
    # Pickle hierarchy_mapping_per_objects
    cPickle.dump(hierarchy_mapping_per_objects, hierarchy_mapping_file, protocol=cPickle.HIGHEST_PROTOCOL)
    # Close the file
    hierarchy_mapping_file.close()
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
    path = os.path.join(TRAINING_OBJECTS_CNN_PATH, time_and_date)
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

    # classes_count, hierarchy_mapping, entities = get_sorted_data(classes_count_file_name="mini_classes_count.p",
    #                                                              hierarchy_mapping_file_name="mini_class_mapping.p",
    #                                                              entities_file_name="final_entities.p",
    #                                                              nof_labels=NOF_LABELS)

    entities, hierarchy_mapping_objects, _ = get_filtered_data(filtered_data_file_name="filtered_module_data.p")

    # Get Visual Genome Data objects
    objects = preprocessing_objects(entities, hierarchy_mapping_objects, object_file_name="filtered_objects.p")

    # If there is too much data tak only part pf the data
    if len(objects) > MAX_NOF_SAMPLES_THR:
        objects = objects[:MAX_NOF_SAMPLES]

    # new_hierarchy_mapping = create_new_hierarchy_mapping(hierarchy_mapping)
    # Get the updating class_mapping and hierarchy_mapping by mapping and save them in Training Folder
    # classes_count, hierarchy_mapping = get_classes_mapping_and_hierarchy_mapping_by_objects(objects, path, config=config)

    train_imgs, test_imgs, val_imgs = splitting_to_datasets(objects, training_percent=TRAINING_PERCENT,
                                                            testing_percent=TESTING_PERCENT, num_epochs=NUM_EPOCHS,
                                                            path=path, config=config)

    # Set the number of classes
    number_of_classes = len(hierarchy_mapping_objects)

    # Create a data generator for VisualGenome
    # data_gen_train_vg = visual_genome_data_cnn_generator(data=train_imgs, hierarchy_mapping=hierarchy_mapping_objects,
    #                                                      config=config, mode='train')
    # data_gen_test_vg = visual_genome_data_cnn_generator(data=test_imgs, hierarchy_mapping=hierarchy_mapping_objects,
    #                                                     config=config, mode='test')
    # data_gen_validation_vg = visual_genome_data_cnn_generator(data=val_imgs, hierarchy_mapping=hierarchy_mapping_objects,
    #                                                           config=config, mode='validation')

    # todo: add batch-size
    data_gen_train_vg = visual_genome_data_cnn_generator_with_batch(data=train_imgs,
                                                                    hierarchy_mapping=hierarchy_mapping_objects,
                                                                    config=config, mode='train', batch_size=NUM_BATCHES)
    data_gen_test_vg = visual_genome_data_cnn_generator_with_batch(data=test_imgs,
                                                                   hierarchy_mapping=hierarchy_mapping_objects,
                                                                   config=config, mode='test', batch_size=NUM_BATCHES)
    data_gen_validation_vg = visual_genome_data_cnn_generator_with_batch(data=val_imgs,
                                                                         hierarchy_mapping=hierarchy_mapping_objects,
                                                                         config=config, mode='validation',
                                                                         batch_size=NUM_BATCHES)

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (config.crop_height, config.crop_width, 3)

    img_input = Input(shape=input_shape_img, name="image_input")

    # Define ResNet50 model With Top
    net = ModelZoo()
    model_resnet50 = net.resnet50_base(img_input, trainable=True)
    model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
    output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
        model_resnet50)

    # Define the model
    model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
    # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
    model.summary()

    # Save the last layer initialized weights
    if config.replace_top:
        last_layer_weights = model.layers[-1].get_weights()

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

    # Replace the last layer
    if config.replace_top:
        # Set the new initialized weights
        model.layers[-1].set_weights(last_layer_weights)

        # Replace the last top layer with a new Dense layer
        # model = replace_top_layer(model, number_of_classes)
        # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
        # model.summary()

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
