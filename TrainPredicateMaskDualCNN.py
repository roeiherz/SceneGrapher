import matplotlib as mpl
from FilesManager.FilesManager import FilesManager

mpl.use('Agg')
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from Data.VisualGenome.models import ObjectMapping, RelationshipMapping
from DesignPatterns.Detections import Detections
from FeaturesExtraction.Lib.VisualGenomeDataGenerator import visual_genome_data_predicate_generator_with_batch, \
    visual_genome_data_predicate_mask_generator_with_batch, visual_genome_data_predicate_mask_dual_generator_with_batch
from FeaturesExtraction.Lib.Zoo import ModelZoo
import os
import cPickle
import numpy as np
from FeaturesExtraction.Lib.Config import Config
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from keras import backend as K
from keras.models import Model
import sys
import matplotlib.pyplot as plt
from FeaturesExtraction.Utils.Utils import get_time_and_date, TRAINING_PREDICATE_MASK_CNN_PATH, \
    replace_top_layer, DATA, \
    VISUAL_GENOME, get_bad_urls
from Utils.Utils import create_folder
from FeaturesExtraction.Utils.data import splitting_to_datasets, process_to_detections, get_filtered_data, \
    get_name_from_file, create_negative_relations, pickle_dataset
from Utils.Logger import Logger

NOF_LABELS = 150
TRAINING_PERCENT = 0.75
VALIDATION_PERCENT = 0.05
TESTING_PERCENT = 0.2
NUM_EPOCHS = 90
NUM_BATCHES = 128
RATIO = 3.0 / 10

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

    objects_path_token = "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(object_file_name))

    # Check if pickles are already created
    objects_path = filemanager.get_file_path(objects_path_token)

    if os.path.isfile(objects_path):
        logger.log('File is already exist {0}'.format(objects_path))
        objects = cPickle.load(open(objects_path))
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
        logger.log("Finished img: {}".format(idx))

    # Pickle objects_lst
    objects_array = np.array(objects_lst)
    # Save the objects files to the disk
    filemanager.save_file(objects_path_token, objects_array)

    return objects_array


def preprocessing_relations(img_data, hierarchy_mapping_objects, hierarchy_mapping_predicates,
                            relation_file_name='full_relations.p', add_negatives=False):
    """
    This function takes the img_data and create a full object list that contains ObjectMapping class
    :param relation_file_name: relation pickle file name
    :param img_data: list of entities files
    :param hierarchy_mapping_objects: dict of objects hierarchy_mapping
    :param hierarchy_mapping_predicates: dict of predicates hierarchy_mapping
    :param add_negatives: Do we want to add negatives to the data
    :return: list of RelationshipMapping
    """

    relations_path_token = "{0}.{1}.{2}".format(DATA, VISUAL_GENOME, get_name_from_file(relation_file_name))

    # Check if pickles are already created
    relations_path = FilesManager().get_file_path(relations_path_token)

    if os.path.isfile(relations_path):
        Logger().log('File is already exist {0}'.format(relations_path))
        relations = FilesManager().load_file(relations_path_token)
        return relations

    # Get the whole objects from entities
    relations_lst = []
    correct_object_labels = hierarchy_mapping_objects.keys()
    correct_predicates_labels = hierarchy_mapping_predicates.keys()
    bad_urls = get_bad_urls()
    # Index for loop
    idx = 0
    # The new negative relations ID
    relation_id = 10000000
    # Counter for negatives
    total_negatives = 0
    # Counter for positives
    total_positives = 0
    for img in img_data:

        # Sorting bad urls
        if img.image.url in bad_urls:
            continue

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
            relations_lst.append(new_relation_mapping)

        total_positives += len(relations)

        # Check the relationship_filtered list is not empty
        if add_negatives and relations_lst:
            # Create Negatives
            negative_relations, _, relation_id = create_negative_relations(img, relation_id, relation_id,
                                                                           positive_negative_ratio=RATIO)
            # Rewrite relationships
            relations_lst += negative_relations
            total_negatives += len(negative_relations)
            # Print
            Logger().log("Number of (negatives, positive) relations ({0}, {1}) in Entity number: {2}".format(
                len(negative_relations), len(relations), img.image.id))
            Logger().log("Total relations {0} in iter {1}".format(len(relations_lst), idx + 1))
        else:
            Logger().log("Warning: No relations in Entity: {}".format(img.image.id))

        idx += 1
        Logger().log("Finished img: {}".format(img.image.id))

    Logger().log(
        "Number of negatives which have been created: {0}, and number of positives are:{1}".format(total_negatives,
                                                                                                   total_positives))

    # Pickle objects_lst
    relations_array = np.array(relations_lst)

    # Save the objects files to the disk
    FilesManager().save_file(relations_path_token, relations_array)

    return relations_array


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


def pick_different_negative_sample_ratio(detections, ratio=1):
    """
    This function take the detection and decides on a different negative ratio
    :param ratio: the ratio between positive to negative
    :param detections: a Detections dtype array
    :return: 
    """
    # Get positive indices
    pos_indices = np.where(detections[Detections.Predicate] != u'neg')[0]
    # Get negative indices
    neg_indices = np.where(detections[Detections.Predicate] == u'neg')[0]
    # Shuffle randomly negative indices
    np.random.shuffle(neg_indices)
    # Take only the wanted negative ratio
    nof_positive = len(detections) - len(neg_indices)
    chosen_indices = neg_indices[:int(nof_positive * ratio)]
    # Append the positive and the negative indices
    all_indice = np.append(pos_indices, chosen_indices)
    return detections[all_indice]


def sorting_urls(train_imgs, test_imgs, val_imgs):
    """
    This function sorting bad urls from the objects data-sets
    :param train_imgs: train data
    :param test_imgs: test data
    :param val_imgs: validation data
    :return: train, test and validation object list after sorting
    """

    # Get the bad urls
    bad_urls = get_bad_urls()

    if len(bad_urls) < 100:
        logger.log("WARNING: number of bad urls is lower than 100")

    # Remove bad urls
    # Get indices that are not bad urls
    train_indices = np.where(np.in1d(train_imgs[Detections.Url], bad_urls) == False)[0]
    real_train_imgs = train_imgs[train_indices]

    # Get indices that are not bad urls
    test_indices = np.where(np.in1d(test_imgs[Detections.Url], bad_urls) == False)[0]
    real_test_imgs = test_imgs[test_indices]

    # Get indices that are not bad urls
    if len(val_imgs) != 0:
        val_indices = np.where(np.in1d(val_imgs[Detections.Url], bad_urls) == False)[0]
        real_val_imgs = val_imgs[val_indices]
    else:
        real_val_imgs = []

    logger.log("Debug printing after sorting- the number of train samples: {0}, the number of test samples: {1}, "
               "the number of validation samples: {2}".format(len(real_train_imgs),
                                                              len(real_test_imgs),
                                                              len(real_val_imgs)))
    return real_train_imgs, real_test_imgs, real_val_imgs


def get_size_of_detections_testset(detections_test, size_of_test):
    """
    This function returns detections test set according to a specific size
    :param detections_test: the detections test
    :param size_of_test: the wanted test-set size
    :return: 
    """
    detections_test_id = list(detections_test[Detections.Id])
    np.random.shuffle(detections_test_id)
    detections_test = detections_test[np.in1d(detections_test[Detections.Id], detections_test_id[:size_of_test])]
    return detections_test


if __name__ == '__main__':

    # Define FileManager
    filemanager = FilesManager()
    # Define Logger
    logger = Logger()

    # Get argument
    if len(sys.argv) < 2:
        # Default GPU number
        gpu_num = 0
    else:
        # Get the GPU number from the user
        gpu_num = sys.argv[1]

    # Printing which GPU you have selected
    logger.log("Selected GPU number: {0}".format(gpu_num))

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
    path = os.path.join(TRAINING_PREDICATE_MASK_CNN_PATH, time_and_date)
    # Create a new folder for training
    create_folder(path)
    # loading model weights
    if config.loading_model:
        net_weights = filemanager.get_file_path(config.loading_model_token)
        logger.log("Loading Weights from: {}".format(net_weights))
    else:
        # The Weights for training
        net_weights = filemanager.get_file_path(config.base_net_weights)
        logger.log("Taking Base Weights from: {}".format(net_weights))
    net_weights_path = os.path.join(path, config.model_weights_name)
    logger.log("The new Model Weights will be Saved: {}".format(net_weights_path))

    # entities, hierarchy_mapping_objects, hierarchy_mapping_predicates = get_filtered_data(
    #                                                                         filtered_data_file_name=
    #                                                                         "full_filtered_data",
    #                                                                         # "mini_filtered_data",
    #                                                                         category='entities',
    #                                                                         load_entities=False)

    # Get the predicate hierarchy mapping and the number of the predicated classes
    hierarchy_mapping_objects = filemanager.load_file("data.visual_genome.hierarchy_mapping_objects")
    hierarchy_mapping_predicates = filemanager.load_file("data.visual_genome.hierarchy_mapping_predicates")
    # # Load filtered data
    # entities_train = filemanager.load_file("data.visual_genome.full_filtered_preprocessed_data_train")
    # entities_test = filemanager.load_file("data.visual_genome.full_filtered_preprocessed_data_test")

    if config.only_pos and "neg" in hierarchy_mapping_predicates:
        # Remove negative label from hierarchy_mapping_predicates because we want to train only positive
        hierarchy_mapping_predicates.pop("neg")
        RATIO = 0

    # # Get Visual Genome Data relations
    # relations_train = preprocessing_relations(entities_train, hierarchy_mapping_objects, hierarchy_mapping_predicates,
    #                                           relation_file_name="full_relations_train",
    #                                           add_negatives=not config.only_pos)
    # relations_test = preprocessing_relations(entities_test, hierarchy_mapping_objects, hierarchy_mapping_predicates,
    #                                          relation_file_name="full_relations_test",
    #                                          add_negatives=not config.only_pos)

    # Process relations to numpy Detections dtype
    detections_train = process_to_detections(None, detections_file_name="full_detections_train")
    detections_test = process_to_detections(None, detections_file_name="full_detections_test")

    logger.log('Number of train detections before sorting negatives: {0} '
               'and test detections after sorting negatives {1}'.format(len(detections_train), len(detections_test)))

    # Get new negative - positive ratio
    detections_train = pick_different_negative_sample_ratio(detections_train, ratio=RATIO)
    detections_test = pick_different_negative_sample_ratio(detections_test, ratio=RATIO)
    size_of_test = len(detections_train) / 3
    detections_test = get_size_of_detections_testset(detections_test, size_of_test)
    # No validation test
    detections_val = []

    # # Split the data to train, test and validate
    # train_imgs, test_imgs, val_imgs = splitting_to_datasets(detections, training_percent=TRAINING_PERCENT,
    #                                                         testing_percent=TESTING_PERCENT, num_epochs=NUM_EPOCHS,
    #                                                         path=path, config=config)

    # Sorting bad urls - should be delete sometime
    train_imgs, test_imgs, val_imgs = sorting_urls(detections_train, detections_test, detections_val)
    # Save train-set and test-set and validation-set
    pickle_dataset(train_imgs, test_imgs, val_imgs, path)

    # Create a data generator for VisualGenome with batch size
    data_gen_train_vg = visual_genome_data_predicate_mask_dual_generator_with_batch(data=train_imgs,
                                                                                    hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                    config=config, mode='train',
                                                                                    classification=Detections.Predicate,
                                                                                    type_box=Detections.UnionBox,
                                                                                    batch_size=NUM_BATCHES)

    # Create a data generator for VisualGenome
    data_gen_test_vg = visual_genome_data_predicate_mask_dual_generator_with_batch(data=test_imgs,
                                                                                   hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                   config=config, mode='test',
                                                                                   classification=Detections.Predicate,
                                                                                   type_box=Detections.UnionBox,
                                                                                   batch_size=NUM_BATCHES)

    # Create a data generator for VisualGenome
    data_gen_validation_vg = visual_genome_data_predicate_mask_dual_generator_with_batch(data=val_imgs,
                                                                                         hierarchy_mapping=hierarchy_mapping_predicates,
                                                                                         config=config, mode='valid',
                                                                                         classification=Detections.Predicate,
                                                                                         type_box=Detections.UnionBox,
                                                                                         batch_size=NUM_BATCHES)

    # Set the number of classes
    if config.replace_top:
        number_of_classes = config.nof_classes
    else:
        number_of_classes = len(hierarchy_mapping_predicates)

    if K.image_dim_ordering() == 'th':
        input_shape_img = (5, None, None)
    else:
        input_shape_img = (config.crop_height, config.crop_width, 5)

    img_input = Input(shape=input_shape_img, name="image_input")

    # Define ResNet50 model Without Top
    net = ModelZoo()
    model_resnet50 = net.resnet50_with_masking_dual(img_input, trainable=True)
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
            logger.log('loading weights from {}'.format(net_weights))
            model.load_weights(net_weights, by_name=True)
    except Exception as e:
        logger.log('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        raise Exception(e)

    # Replace the last layer
    if config.replace_top:
        # Set the new initialized weights
        # model.layers[-1].set_weights(last_layer_weights)

        # Replace the last top layer with a new Dense layer
        model = replace_top_layer(model, len(hierarchy_mapping_predicates))
        # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
        model.summary()

    optimizer = Adam(1e-7)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ModelCheckpoint(net_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
                 TensorBoard(log_dir="logs", write_graph=True, write_images=True),
                 CSVLogger(os.path.join(path, 'training.log'), separator=',', append=False),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00000001)]

    logger.log('Starting training')
    history = model.fit_generator(data_gen_train_vg, steps_per_epoch=len(train_imgs) / NUM_BATCHES, epochs=NUM_EPOCHS,
                                  validation_data=data_gen_test_vg, validation_steps=len(test_imgs) / NUM_BATCHES,
                                  callbacks=callbacks, max_q_size=1, workers=1)

    # Validating the model
    test_score = model.evaluate_generator(data_gen_validation_vg, steps=len(val_imgs) / NUM_BATCHES, max_q_size=1,
                                          workers=1)
    # Plot the Score
    logger.log("The Validation loss is: {0} and the Validation Accuracy is: {1}".format(test_score[0], test_score[1]))

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
