from App.Controller.PumpLight import PumpLight
from FilesManager.FilesManager import FilesManager
from Utils.Logger import Logger

__author__ = 'roeih'

if __name__ == '__main__':
    # Define FileManager
    filemanager = FilesManager()
    # Define Logger
    logger = Logger()

    pl = PumpLight()
