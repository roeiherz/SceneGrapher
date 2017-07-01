import os
import os.path
import sys
sys.path.append("..")
from DesignPatterns.Singleton import Singleton
import cPickle
import yaml

FILE_MANAGER_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_MANAGER_FILENAME = "files.yaml"

class FilesManager(object):
    """
    Files Manager used to load and save any kind of files
    """
    __metaclass__ = Singleton

    def __init__(self, overrides_filename=None):
        """
        Constructor for FilesManager
        :param overrides_filename: "*.yaml file used to override paths to files
        """
        # save input data
        self.overrides_filename = overrides_filename

        # load file paths
        stream = file(os.path.join(FILE_MANAGER_PATH, FILE_MANAGER_FILENAME), 'r')
        self.files = yaml.load(stream)

        # override file paths
        # TBD

    def load_file(self, tokens):
        """
        load file given file tokens
        :param tokens: tokens delimited with '.'
        :return: data read from pickle file
        """
        # get file path
        fileinfo = self.get_file_info(tokens)
        filetype = fileinfo["type"]
        filename = fileinfo["name"]

        # load data per file type
        if filetype == "pickle":
            picklefile = open(fileinfo["name"], "rb")

            # get number of objects stored in the pickle file
            nof_objects = 1
            if "nof_objects" in fileinfo:
                nof_objects = fileinfo["nof_objects"]

            # load data
            if nof_objects == 1:
                data = cPickle.load(picklefile)
            else:
                data = []
                for i in range(nof_objects):
                    data.append(cPickle.load(picklefile))

            picklefile.close()
            return data

        elif filetype == "text":
            with open(filename) as f:
                lines = f.readlines()
                return lines


    def save_file(self, tokens, data):
        """
        save file given tokens in pickle format
        :param tokens: tokens delimited with '.'
        :param data: data to save
        :return: void
        """
        # get file path
        fileinfo = self.get_file_info(tokens)

        # load data
        picklefile = open(fileinfo["name"], "wb")
        # get number of objects stored in the pickle file
        nof_objects = 1
        if "nof_objects" in fileinfo:
            nof_objects = fileinfo["nof_objects"]
        if nof_objects == 1:
            cPickle.dump(data, picklefile)
        else:
            for elem in data:
                cPickle.dump(elem, picklefile)

        picklefile.close()

    def file_exist(self, tokens):
        """
        check if file exists given tokens
        :param tokens: tokens delimited with '.'
        :return: True if file exist
        """
        # get file path
        fileinfo = self.get_file_info(tokens)

        return os.path.exists(fileinfo["name"])


    def get_file_info(self, tokens):

        """
        get file name given file tokens
        :param tokens: tokens delimited with '.'
        :return: path to file
        """
        # get list of tokens
        tokens = tokens.split(".")

        # get filename
        fileinfo = self.files
        for token in tokens:
            if fileinfo.has_key(token):
                fileinfo = fileinfo[token]
            else:
                raise Exception("unknown name token {0} for name {1}", token, tokens)

        # make sure fileinfo was extracted
        if not "name" in fileinfo:
            raise Exception("uncomplete file tokens", tokens)

        return fileinfo