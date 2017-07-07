import os
import os.path
import numpy as np
import sys
from FilesManager.FilesManager import FilesManager
import time


sys.path.append("..")
from DesignPatterns.Singleton import Singleton
import cPickle


class Logger(object):
    """
    Word to vector embedding, using GLOVE
    """
    __metaclass__ = Singleton

    def __init__(self, name="", path=None):
        """
        Creating logger, the name of the logger will be printed at each line and log will be saved in path
        :param name: name of the logger
        :param path: path to log file
        """
        self.name = name
        if self.name != "":
            self.prefix = self.name + ": "
        else:
            self.prefix = ""

        self.path = path
        # create dir
        if self.path != None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            # create file to log
            self.log_file = open(self.path + "/logger.log", "w")
        else:
            self.log_file = None

        # add logger to file manager
        filesmanager = FilesManager()
        filesmanager.add_logger(self)

        # default log file
        if self.log_file == None:
            self.path = filesmanager.get_file_path("logs")
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            self.log_file = open(self.path + "/logger-%s.log" % time.strftime("%c"), "w")
            self.log("Start %s" % time.strftime("%c"))

    def log(self, str):
        if self.log_file != None:
            self.log_file.write(self.name + ": " + str + "\n")
            self.log_file.flush()
        print(self.prefix + str)

    def get_dir(self):
        return self.path
