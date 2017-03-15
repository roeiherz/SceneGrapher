import os

from keras.layers import GlobalAveragePooling2D, Dense, Convolution2D

from Trax.Algo.Deep.Core.GlobalResources import NETWORK_SECTION, NETWORK_CROP_WIDTH, NETWORK_CROP_HEIGHT, \
    DATA_SET_SECTION, APPLICATION_SECTION, APPLICATION_NAME, NETWORK_FOLDER_NAME, DATA_SET_PROJECT_NAME_KEY, \
    getNetworksFolder
from Trax.Miscellaneous.Users.Itsik.TensorFlow.SimpleCNN import train
from Trax.Miscellaneous.Users.Itsik.TensorFlow.Zoo import ModelZoo
from Trax.Miscellaneous.Users.Itsik.TensorFlow.Global import load_model
from Trax.Utils.Conf.Configuration import Config
from Trax.Utils.Logging.Logger import Log

if __name__ == '__main__':
    Config().init()
    Log().init('TensorFlow')

    crop_width = Config.confDict[NETWORK_SECTION][NETWORK_CROP_WIDTH]
    crop_height = Config.confDict[NETWORK_SECTION][NETWORK_CROP_HEIGHT]
    project = Config.confDict[DATA_SET_SECTION][DATA_SET_PROJECT_NAME_KEY]
    app_name = Config.confDict[APPLICATION_SECTION][APPLICATION_NAME]
    base_network = Config.confDict[APPLICATION_SECTION][NETWORK_FOLDER_NAME]

    input_shape = (crop_height, crop_width, 3)

    base_path = getNetworksFolder()

    base_network_folder = os.path.join(base_path, project, app_name, base_network)
    _networks = os.listdir(base_network_folder)
    if _networks:

        Log.info('Loading network {0} from {1}'.format(_networks[0], base_network_folder))

        model = load_model(os. path.join(base_network_folder))
        model.pop()
        model.pop()

        # Switch layer to be untrainable
        for layer in model.layers:
            layer.trainable = False
    else:
        model = ModelZoo().vgg19(convolution_only=True)
        model.add(Convolution2D(2048, 1, 1, activation='relu'))
        model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu'))

    train(model)
