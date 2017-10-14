from App.Controller.AppController import AppController
from App.Model.AppModel import AppModel
from Data.VisualGenome.local import GetAllImageData, GetSceneGraph
from FeaturesExtraction.Utils.Utils import TRAINING_OBJECTS_CNN_PATH, WEIGHTS_NAME, TRAINING_PREDICATE_MASK_CNN_PATH, \
    PROJECT_ROOT, VG_DATA_PATH, get_time_and_date
import getpass
import os

from Utils.Utils import create_folder

__author__ = 'roeih'


class PumpLight(object):
    """
    This class represents a PumpLight app
    """

    FEATURE_EXTRACTOR_FOLDER = "Feature_Extractor_Module_Output"
    RNN_BELIEF_FOLDER = "RNN_Belief_Module_Output"

    def __init__(self):
        """
        This function initializing PumpLight Object
        """

        # Start the interactive messages
        usr_name = getpass.getuser()
        print("HELLO {0}".format(usr_name))
        print("WELCOME TO SCENE GRAPHER \n")

        # The field will hold the model object
        self.model = False
        # The field will hold the model object
        self.controller = False
        # The field use the information if we are still running
        self.running = True
        # The output path
        # Get time and date
        self.time_and_date = get_time_and_date()
        self.output_path = None
        # Get the whole images string dict
        self.images = {img.id: img for img in GetAllImageData(os.path.join(PROJECT_ROOT, VG_DATA_PATH))}
        # Get the whole images ids string list
        self.images_id = self.images.keys()

        # Start
        self.run()

    def run(self):
        """
        This function
        :return:
        """

        print ("Enter the following details: \n")

        # Get image ID
        self.img_id = self.get_img_id()
        # todo: debug
        # self.img_id = 2338129

        # Get GPU number
        self.gpu_num = self.get_gpu_num()

        # Get Objects weights
        # todo: delete inplace objects_dir
        # objects_dir = raw_input('Objects networks directory name: ')
        objects_dir = "Sat_Sep_16_18:36:19_2017"
        objects_model_weight_path = os.path.join(PROJECT_ROOT, TRAINING_OBJECTS_CNN_PATH, objects_dir, WEIGHTS_NAME)
        if not os.path.exists(objects_model_weight_path):
            print("Error: No objects weights exists in {0}".format(objects_model_weight_path))
            return

        # todo: delete inplace predicates_dir
        # Get Predicates weights
        # predicates_dir = raw_input('Predicates networks directory name: ')
        predicates_dir = "Mon_Sep_25_17:47:17_2017"
        predicates_model_weight_path = os.path.join(PROJECT_ROOT, TRAINING_PREDICATE_MASK_CNN_PATH, predicates_dir,
                                                    WEIGHTS_NAME)
        if not os.path.exists(predicates_model_weight_path):
            print("Error: No objects weights exists in {0}".format(predicates_model_weight_path))
            return

        # Get the Model
        self.model = AppModel(objects_model_weight_path, predicates_model_weight_path, self.gpu_num)

        # Get the outputs path
        self.output_path = raw_input('Output directory path: ')
        # todo: debug
        # self.output_path = "/home/roeih/SceneGrapher/Outputs"

        self.output_path = os.path.join(self.output_path, self.time_and_date)
        # Create folder for outputs path if its not exist
        create_folder(self.output_path)

        # Get the Controller
        self.controller = AppController()

        # The options
        while self.running:
            option = raw_input(
                'What can I do for you? [start, draw gt scene graph [gt_sg], choose different image id [img_id], exit] ')
            self.running = self.handle_input(option)

        print('Bye Bye :)')

    def handle_input(self, option):
        """
        This function will handle the input from the user
        :param option: the option
        :return:
        """

        if option.lower() == 'start':
            self.start()
        elif option.lower() == 'gt_sg' or option.lower() == 'draw gt scene graph':
            self.draw_original_gt_scene_graph()
        elif option.lower() == 'img_id' or option.lower() == 'choose different image id':
            self.img_id = self.get_img_id()
        elif option.lower() == 'exit':
            return False
        else:
            print('Command not supported')

        return True

    def start(self):
        start = True
        # The options
        while start:
            print('\n-----INSTRUCTIONS-----'
                  '\nTo start predicting, there are 2 different options: '
                  '\nusing a pre-trained first Module and predict only the second Module (Semi Prediction) or '
                  '\npredicting the first Module and second Module together (Full Prediction).'
                  '\n\nFull Prediction: Predict Feature Extraction Module and then RNN Belief Module. '
                  '\nSemi Prediction: Load a predicted Feature Extraction Module and just Predict RNN Belief Module. '
                  '\n -----END OF INSTRUCTIONS-----\n')
            option = raw_input('What can I do want to do? [full prediction [full], semi prediction [semi], back] ')
            start = self.handle_start_input(option)

    def handle_start_input(self, option):
        """
        This function will handle the start input from the user
        :param option: the option
        :return:
        """
        if option.lower() == 'full prediction' or option.lower() == 'full':
            self.handle_prediction(use_full_predict=True)
        elif option.lower() == 'semi prediction' or option.lower() == 'semi':
            self.handle_prediction(use_full_predict=False)
        elif option.lower() == 'back':
            return False
        else:
            print('command not supported')

        return True

    def handle_prediction(self, use_full_predict=False):
        """

        :param use_full_predict:
        :return:
        """
        start = True
        # The options
        while start:
            print('\n-----INSTRUCTIONS-----'
                  '\nChoose from which model do you wish to see outputs. '
                  '\nChoose *1* for Feature Extraction Module or *2* for RNN Belief Module. '
                  '\n -----END OF INSTRUCTIONS-----\n')
            option = raw_input('Choose model: [Feature Extraction Module [1], RNN Belief Module [2], back] ')
            start = self.handle_full_prediction_input(option, use_full_predict)

    def handle_full_prediction_input(self, option, use_full_predict):
        """
        This function will handle the full prediction input from the user
        :param use_full_predict:
        :param option: the option
        :return:
        """
        if option.lower() == '1':
            self.predict(use_full_predict=use_full_predict, rnn_module_output=False)
        elif option.lower() == '2':
            self.predict(use_full_predict=use_full_predict, rnn_module_output=True)
        elif option.lower() == 'back':
            return False
        else:
            print('command not supported')

        return True

    def predict(self, use_full_predict=False, rnn_module_output=True):
        """
        This function do the prediction full-prediction (Feature Extraction module + Belief RNN module)
        or a semi-prediction (only Belief RNN module)
        :param rnn_module_output: True: the outputs that you will see is from the Belief RNN module.
                                 False: the outputs that you will see is from the Feature Extraction module.
        :param use_full_predict: True: means full-prediction (Feature Extraction module + Belief RNN module).
                                 False: semi-prediction (only Belief RNN module)
        :return: True or False
        """

        # Get the hierarchy_mappings of Objects and Predicates
        objects_hierarchy_mapping = self.controller.get_objects_hierarchy_mapping()
        predicates_hierarchy_mapping = self.controller.get_predicates_hierarchy_mapping()

        # Check Hierarchy Mappings are not None
        if objects_hierarchy_mapping is not None:
            number_of_classes_objects = len(objects_hierarchy_mapping)

        if predicates_hierarchy_mapping is not None:
            number_of_classes_predicates = len(predicates_hierarchy_mapping)

        # Do we need to load the Feature Extraction module for Prediction
        if use_full_predict:
            # Get Entity
            entity = self.controller.load_entity(self.img_id)

            if entity is None:
                print("Entity has not been found. Go back and choose another Image ID.")
                return False

            # Load Feature Extraction Module for Prediction
            self.model.load_feature_extractions_model(number_of_classes_objects, number_of_classes_predicates)

            # Get the url data for image
            url_data = entity.image.url

            # Create Objects Mapping type
            objects = self.controller.from_object_to_objects_mapping(entity.objects, objects_hierarchy_mapping,
                                                                     url_data)

            if len(objects) == 0:
                print("No Objects have been found")
                return False

            # Predict objects per entity
            self.model.predict_objects_for_module(entity, objects, url_data, objects_hierarchy_mapping)

            # Predict predicates per entity
            self.model.predict_predicates_for_module(entity, objects, url_data, predicates_hierarchy_mapping,
                                                     use_mask=True)
        else:
            # Load pickled entity after earlier prediction
            entity = self.controller.load_preprocessed_entity(self.img_id)
            # Plot stats on the pre-trained entity
            self.controller.plot_stats(entity)

        # Load the Belief RNN Model
        self.model.load_belief_rnn_model(number_of_classes_objects, number_of_classes_predicates)

        if rnn_module_output:
            # Run Prediction of the Belief RNN Model
            self.model.predict_rnn_belief_module(entity)
            # Plot stats on the pre-trained entity
            self.controller.plot_stats(entity)

        start = True
        # The options
        while start:
            option = raw_input(
                'What can I do want to do? [draw objects [obj], draw relationships [rel], draw scene graph [sg] ,back]')
            start = self.handle_predict_input(option, entity, rnn_module_output)

    def handle_predict_input(self, option, entity, rnn_module_output):
        """
        This function will handle the predict input from the user
        :param rnn_module_output: Used for saving the images per different folder (depends the choosing of rnn_module_output)
                                 True: the outputs that you will see is from the Belief RNN module.
                                 False: the outputs that you will see is from the Feature Extraction module.
        :param entity: entity Visual Genome type
        :param option: the option
        :return: True or False
        """

        # Get the current output path per image id and module directory to output
        save_path = self.get_curr_path(rnn_module_output)
        # Create folder for outputs path if its not exist
        create_folder(save_path)

        if option.lower() == 'draw objects' or option.lower() == 'obj':
            self.controller.draw_objects(entity, save_path)
        elif option.lower() == 'draw relationships' or option.lower() == 'rel':
            self.controller.draw_relationships(entity, save_path)
        elif option.lower() == 'draw scene graph' or option.lower() == 'sg':
            self.controller.draw_prediction_scene_graph(entity, save_path)
        elif option.lower() == 'back':
            return False
        else:
            print('command not supported')

        return True

    def get_curr_path(self, rnn_module_output=None):
        """
        This function set the current path per image_id and rnn_module_output
        :param rnn_module_output: Used for saving the images per different folder (depends the choosing of rnn_module_output)
                                 True: the outputs that you will see is from the Belief RNN module.
                                 False: the outputs that you will see is from the Feature Extraction module.
        :return: the output directory path
        """
        if rnn_module_output is None:
            return os.path.join(self.output_path, str(self.img_id))

        if rnn_module_output:
            return os.path.join(self.output_path, str(self.img_id), self.RNN_BELIEF_FOLDER)

        if not rnn_module_output:
            return os.path.join(self.output_path, str(self.img_id), self.FEATURE_EXTRACTOR_FOLDER)

    def get_img_id(self):
        """
        This function get Image ID
        :return: int of Image ID
        """
        # Get image ID
        img_id = int(raw_input('Choose Image ID: '))
        while img_id not in self.images_id:
            img_id = int(raw_input("Wrong Image ID. please enter image ID again: "))
        print("The image ID that was seleted is {0} in url: {1}".format(img_id, self.images[img_id].url))
        return img_id

    def draw_original_gt_scene_graph(self):
        """
        This function draw the original ground-truth scene graph without any sorting the data (objects and predicates)
        before.
        """
        entity = GetSceneGraph(self.img_id, images=self.images,
                               imageDataDir=os.path.join(PROJECT_ROOT, VG_DATA_PATH, 'by-id/'),
                               synsetFile=os.path.join(PROJECT_ROOT, VG_DATA_PATH, 'synsets.json'))

        save_path = os.path.join(self.output_path, str(self.img_id))
        create_folder(save_path)
        self.controller.draw_original_gt_scene_graph(entity, save_path)

    def get_gpu_num(self):
        """
        This function get GPU number from the user and set it to CUDA VISIBLE DEVICES
        :return: GPU number
        """

        # Get GPU number
        gpu_num = raw_input('Please insert number of GPU to be used: ')

        # Define GPU training
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

        return gpu_num

