import ast
import os

import cPickle

import numpy
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, ProgbarLogger, LearningRateScheduler
from keras.layers import Dense, functools
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from Trax.Algo.Deep.Core.Datasets.Builder import BuildClassMapping
from Trax.Algo.Deep.Core.Datasets.Dataset import PredictionDataset, TrainDataset
from Trax.Algo.Deep.Core.Datasets.Preprocessors.Jitter import MultiJitter
from Trax.Algo.Deep.Core.Datasets.Preprocessors.Krizhevsky import TrainingPreprocessor, PredictionPreprocessor
from Trax.Algo.Deep.Core.Global import LogglyLogger, HIERARCHY_MAPPING_FILE_NAME
from Trax.Algo.Deep.Core.GlobalResources import DATA_SET_SECTION, HYPER_PARAMS_SECTION, generateOutputFolderPath, \
    APPLICATION_SECTION, DATA_SET_PROJECT_NAME_KEY, APPLICATION_NAME, TRAINING_OPTIMIZATION_SECTION, getNetworksFolder, \
    NETWORK_SECTION, NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT, NETWORK_CROP_WIDTH, NETWORK_CROP_HEIGHT, \
    NETWORK_PADDING_METHOD, JITTERS, PREDICTION_DATA_SET_NAME_KEY, getOutputFolderPath, TRAIN_DATA_SET_NAME_KEY, \
    MULTI_PROJECT, NETWORK_FOLDER_NAME, SAMPLES, LABELS
from Trax.Algo.Deep.Core.Utils.PathMapper import map_path
from Trax.Utils.Conf.Configuration import Config
from Trax.Utils.Files.FilesServices import create_folder
from Trax.Utils.Logging.Logger import Log

SAMPLES_PER_EPOCH = 150000
NUMBER_OF_EPOCHS = 100


# def generator(data_set, preprocessor):
#     def generate():
#         for batch in data_set.get_eager_iterator(preprocessor):
#             yield batch[SAMPLE], batch[LABEL]
#
#     return create_generator

def schedual(epoch, learning_rate, epoch_list):
    if epoch in epoch_list:
        return learning_rate * 10 ^ (-1 * epoch_list.index(epoch))
    else:
        return learning_rate


class IteratorAdapter():
    def __init__(self, dataset, preprocessor, classes_number):
        self._dataset = dataset
        self._preprocessor = preprocessor
        self._inner_iterator = None
        self._classes_number = classes_number

    def __iter__(self):
        return self

    def next(self):
        if self._inner_iterator is None:
            self._inner_iterator = self._dataset.get_eager_iterator(self._preprocessor)

        try:
            batch = self._inner_iterator.next()
        except StopIteration as e:
            self._inner_iterator = self._dataset.get_eager_iterator(self._preprocessor)
            batch = self._inner_iterator.next()

        # Convert to TF format
        return batch[SAMPLES].transpose(0, 3, 2, 1), np_utils.to_categorical(batch[LABELS], self._classes_number)

    def __len__(self):
        return self._classes_number


def learning_rate_scheduler(base_learning_rate, epoch_scheduling, epoch_number):
    if epoch_number in epoch_scheduling:
        return base_learning_rate * 0.1 ^ (epoch_scheduling.index(epoch_number) + 1)


class LogEpochResults(Callback):
    def __init__(self):
        self._loggly_logger = LogglyLogger()

    def on_batch_end(self, batch, logs=None):

        for key, value in logs.iteritems():
            if type(value) == numpy.float32:
                logs[key] = float(value)

        logs['Type'] = 'batch'

        self._loggly_logger.log(logs)

    def on_epoch_end(self, epoch, logs=None):

        for key, value in logs.iteritems():
            if type(value) == numpy.float32:
                logs[key] = float(value)

        logs['Type'] = 'epoch'

        self._loggly_logger.log(logs)


def importHierarchyMapping(hierarchy_mapping_path):
    if os.path.isdir(hierarchy_mapping_path):
        hierarchy_mapping_path = os.path.join(hierarchy_mapping_path, HIERARCHY_MAPPING_FILE_NAME)

    Log.info('Unpickling hierarchy_mapping from {0}'.format(hierarchy_mapping_path))

    with file(hierarchy_mapping_path, 'rb') as hierarchy_mapping_file:
        unpickler = cPickle.Unpickler(hierarchy_mapping_file)
        unpickler.find_global = map_path
        hierarchy_mapping = unpickler.load()

    return hierarchy_mapping


def loadDataset(data_set_params, hyper_parameters):
    train_batch_size = hyper_parameters['train_batch_size']
    test_batch_size = hyper_parameters['test_batch_size']
    project = Config.confDict[DATA_SET_SECTION][DATA_SET_PROJECT_NAME_KEY]
    app_name = Config.confDict[APPLICATION_SECTION][APPLICATION_NAME]
    folder = Config.confDict[APPLICATION_SECTION][NETWORK_FOLDER_NAME]

    if MULTI_PROJECT in Config.confDict[DATA_SET_SECTION] and Config.confDict[DATA_SET_SECTION][
        'label_type'] == 'object_code_label':
        projects = ast.literal_eval(Config.confDict[DATA_SET_SECTION][MULTI_PROJECT])
    else:
        projects = None

    train_data_set = None
    prediction_data_set = None

    negative_limit = data_set_params.get('negative_limit', 0)
    positive_limit = data_set_params.get('positive_limit', 0)

    if TRAIN_DATA_SET_NAME_KEY in data_set_params:
        hierarchyMapping = BuildClassMapping(data_set_params['project'], data_set_params['label_type']).build(
            projects=projects)
        train_data_set = TrainDataset(data_set_params[DATA_SET_PROJECT_NAME_KEY],
                                      data_set_params[TRAIN_DATA_SET_NAME_KEY],
                                      train_batch_size,
                                      test_batch_size,
                                      data_set_params['label_type'],
                                      hierarchyMapping,
                                      data_set_params['brand_none_negatives'],
                                      data_set_params['product_none_negatives'],
                                      data_set_params['form_factor_none_negatives'],
                                      data_set_params.get('brand_code_none_negatives', []),
                                      negative_limit,
                                      positive_limit,
                                      data_set_params.get('use_cache', False),
                                      data_set_params.get('cascade', False),
                                      data_set_params.get('validation_percent', 0.05),
                                      data_set_params.get('batch_num_limit', 0))

    if PREDICTION_DATA_SET_NAME_KEY in data_set_params:
        hierarchyMapping = importHierarchyMapping(getOutputFolderPath(project, app_name, folder))
        prediction_data_set = PredictionDataset(data_set_params[DATA_SET_PROJECT_NAME_KEY],
                                                data_set_params[PREDICTION_DATA_SET_NAME_KEY],
                                                test_batch_size,
                                                data_set_params['label_type'],
                                                hierarchyMapping,
                                                data_set_params['brand_none_negatives'],
                                                data_set_params['product_none_negatives'],
                                                data_set_params['form_factor_none_negatives'],
                                                data_set_params.get('brand_code_none_negatives', []),
                                                negative_limit,
                                                positive_limit,
                                                data_set_params.get('use_cache', False),
                                                data_set_params.get('batch_num_limit', 0))

    return train_data_set, prediction_data_set


def train(model):
    # Config params
    data_set_params = Config.confDict[DATA_SET_SECTION]
    hyper_parameters = Config.confDict[HYPER_PARAMS_SECTION]
    project = Config.confDict[DATA_SET_SECTION][DATA_SET_PROJECT_NAME_KEY]
    app_name = Config.confDict[APPLICATION_SECTION][APPLICATION_NAME]
    base_network = Config.confDict[APPLICATION_SECTION][NETWORK_FOLDER_NAME]
    training_optimization_params = Config.confDict[TRAINING_OPTIMIZATION_SECTION]
    jitters_list = Config.confDict[DATA_SET_SECTION].get(JITTERS, None)
    image_width = Config.confDict[NETWORK_SECTION][NETWORK_IMAGE_WIDTH]
    image_height = Config.confDict[NETWORK_SECTION][NETWORK_IMAGE_HEIGHT]
    crop_width = Config.confDict[NETWORK_SECTION][NETWORK_CROP_WIDTH]
    crop_height = Config.confDict[NETWORK_SECTION][NETWORK_CROP_HEIGHT]
    padding_method = Config.confDict[NETWORK_SECTION][NETWORK_PADDING_METHOD]

    # Import dataset
    train_data_set, prediction_data_set = loadDataset(data_set_params, hyper_parameters)

    if jitters_list and len(jitters_list) > 0 and len(jitters_list[0]) > 0:
        jitter = MultiJitter(image_width, image_height, crop_width, crop_height, jitters_list, padding_method)
    else:
        jitter = None

    index_to_label_mapping, label_to_index_mapping = train_data_set.generateMappings()

    training_processor = TrainingPreprocessor(label_to_index_mapping,
                                              jitter,
                                              image_width, image_height,
                                              crop_width, crop_height, padding_method)

    prediction_processor = PredictionPreprocessor(label_to_index_mapping,
                                                  jitter,
                                                  image_width, image_height,
                                                  crop_width, crop_height, padding_method)

    # Serialization output
    base_path = getNetworksFolder()
    tensor_board_output = os.path.join(base_path, project, app_name, 'logs')
    folder_name, folder_path = generateOutputFolderPath(project, app_name)

    file_path = os.path.join(folder_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    create_folder(tensor_board_output)

    # Optimizer
    sgd = SGD(lr=training_optimization_params['base_learning_rate'], momentum=0.9, nesterov=True)

    model.add(Dense(len(index_to_label_mapping), activation='softmax'))

    _schedual = functools.partial(schedual, learning_rate=training_optimization_params['base_learning_rate'],
                                  epoch_list=[20, 30, 40, 50, 60, 90])

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy', 'fmeasure'])
    model.fit_generator(generator=IteratorAdapter(train_data_set.train, training_processor,
                                                  len(index_to_label_mapping)),
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NUMBER_OF_EPOCHS,
                        callbacks=[ModelCheckpoint(filepath=file_path,
                                                   save_best_only=True),
                                   ProgbarLogger(),
                                   TensorBoard(log_dir=tensor_board_output),
                                   LogEpochResults(),
                                   LearningRateScheduler(_schedual)],
                        validation_data=IteratorAdapter(train_data_set.validation, prediction_processor,
                                                        len(index_to_label_mapping)),
                        nb_val_samples=train_data_set.validation_set_size)

