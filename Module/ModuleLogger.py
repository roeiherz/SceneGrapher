import os
import os.path
import numpy as np
import sys
sys.path.append("..")
from DesignPatterns.Singleton import Singleton
import cPickle


class ModuleLogger(object):
    """
    Word to vector embeddin, using GLOVE
    """
    __metaclass__ = Singleton

    def __init__(self, name):
        self.name = name

        # create dir
        if not os.path.exists(name):
            os.makedirs(name)
        else:
            # remove all files in dir
            for the_file in os.listdir(name):
                file_path = os.path.join(name, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

        # create file to log
        self.log_file = open(name + "/logger.log", "w")

    def log(self, str):
        self.log_file.write(self.name + ": " + str + "\n")
        self.log_file.flush()
        print(self.name + ": " + str)

    def get_dir(self):
        return self.name
