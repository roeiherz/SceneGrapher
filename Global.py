import glob
import os
import keras.models


def load_model(folder_path):
    _files = glob.glob(os.path.join(folder_path, '*.hdf5'))

    if len(_files) > 1:
        print("Found more then one network file in path {0}".format(folder_path))

    _network_path = _files[0]
    return keras.models.load_model(_network_path)
