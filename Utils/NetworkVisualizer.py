import os
import cPickle
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from features_extraction.Lib.Config import Config
from features_extraction.Lib.Zoo import ModelZoo
from keras import backend as K
from keras.models import Model
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.visualization import visualize_saliency
from features_extraction.Utils.Utils import WEIGHTS_NAME, VG_VisualModule_PICKLES_PATH

__author__ = 'roeih'


class NetworkVisualizer(object):
    """
    This class is a network visualizer
    """

    def __init__(self, weights_name_dir, hierarchy_mapping_name, gpu_num=0):
        """
        This function initializes network visualizer
        :param weights_name_dir: the path for the weights
        :param hierarchy_mapping_name: the path for the hierarchy mapping
        :param gpu_num: 0 (default)
        """

        # Define the hierarchy mapping path
        hierarchy_mapping_name = os.path.join("..", VG_VisualModule_PICKLES_PATH, hierarchy_mapping_name)

        # Define the model path
        model_path = os.path.join("..", weights_name_dir, WEIGHTS_NAME)

        # Check if weights are declared properly
        if not os.path.exists(model_path) or not os.path.exists(hierarchy_mapping_name):
            print(
                "Error: No Weights have been found or No Hierarchy Mapping has been found in {0} or {1}".format(
                    model_path,
                    hierarchy_mapping_name))
            raise Exception

        # Get the hierarchy mapping
        self.hierarchy_mapping_objects = cPickle.load(open(hierarchy_mapping_name))
        # Set the number of classes of object
        self.nof_classes = len(self.hierarchy_mapping_objects)
        # Load class config
        self.config = Config(gpu_num)
        # Get the model
        self.model = self.get_model(self.nof_classes, weight_path=model_path)

    def get_model(self, number_of_classes, weight_path):
        """
        This function loads the model
        :param weight_path: model weights path
        :type number_of_classes: number of classes
        :return: model
        """

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
        else:
            input_shape_img = (self.config.crop_height, self.config.crop_width, 3)

        img_input = Input(shape=input_shape_img, name="image_input")

        # Define ResNet50 model Without Top
        net = ModelZoo()
        model_resnet50 = net.resnet50_base(img_input, trainable=True)
        model_resnet50 = GlobalAveragePooling2D(name='global_avg_pool')(model_resnet50)
        output_resnet50 = Dense(number_of_classes, kernel_initializer="he_normal", activation='softmax', name='fc')(
            model_resnet50)

        # Define the model
        model = Model(inputs=img_input, outputs=output_resnet50, name='resnet50')
        # In the summary, weights and layers from ResNet50 part will be hidden, but they will be fit during the training
        model.summary()

        # Load pre-trained weights for ResNet50
        try:
            print("Start loading Weights")
            model.load_weights(weight_path, by_name=True)
            print('Finished successfully loading weights from {}'.format(weight_path))

        except Exception as e:
            print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
                'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            ))
            raise Exception(e)

        print('Finished successfully loading Model')
        return model

    def visualize_conv_layers(self, layer_name='conv1', savefig_path=""):
        """
        Each conv layer has several learned 'template matching' filters that maximize their output when a similar
        template pattern is found in the input image
        """

        # The name of the layer we want to visualize
        layer_idx = [idx for idx, layer in enumerate(self.model.layers) if layer.name == layer_name][0]

        # Visualize all filters in this layer.
        filters = np.arange(get_num_filters(self.model.layers[layer_idx]))

        # Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
        vis_images = []
        for idx in filters:
            img = visualize_activation(self.model, layer_idx, filter_indices=idx)
            img = utils.draw_text(img, str(idx))
            vis_images.append(img)

        # Generate stitched image palette with 8 cols.
        stitched = utils.stitch_images(vis_images, cols=8)
        plt.axis('off')
        plt.imshow(stitched)
        plt.title(layer_name)
        plt.savefig(savefig_path)

        print('debug')

    def visualize_dense_layers(self, last_layer='fc', savefig_path="", nof_times=3, class_number=20):
        """
        Generate an input image that maximizes the final Dense layer output corresponding to some class class.
        :param last_layer: last layer name
        :param savefig_path: where to save the figure
        :param class_number: which class number do we want to check (for example class number 30 is some predicate)
        :param nof_times: how many times do you want to check the class
        :return:
        """

        layer_idx = [idx for idx, layer in enumerate(self.model.layers) if layer.name == last_layer][0]

        # Generate three different images of the same output index.
        vis_images = []

        # Print class number (20) * nof_times (3)
        # for idx in [20, 20, 20]:
        for idx in [class_number] * nof_times:
            img = visualize_activation(self.model, layer_idx, filter_indices=idx, max_iter=500)
            img = utils.draw_text(img, str(idx))
            vis_images.append(img)

        stitched = utils.stitch_images(vis_images)
        plt.axis('off')
        plt.imshow(stitched)
        plt.title(last_layer)
        plt.savefig(savefig_path)

        print('debug')

    def visualize_heat_maps(self, image_paths, last_layer='fc', savefig_path=""):
        """
        compute the gradient of output category with respect to input image.
        This should tell us how output category value changes with respect to a small change in input image pixels.
        :param image_paths: a list of images which we want check.
                for example:
                image_paths = ["https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
                "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
                "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"]
        :param last_layer: last layer name
        :param savefig_path: where to save the figure
        :return:
        """

        layer_idx = [idx for idx, layer in enumerate(self.model.layers) if layer.name == last_layer][0]

        heatmaps = []
        for path in image_paths:
            seed_img = utils.load_img(path, target_size=(self.config.crop_height, self.config.crop_width))
            x = np.expand_dims(img_to_array(seed_img), axis=0)
            x = preprocess_input(x)
            pred_class = np.argmax(self.model.predict(x))

            # Here we are asking it to show attention such that prob of `pred_class` is maximized.
            heatmap = visualize_saliency(self.model, layer_idx, [pred_class], seed_img)
            heatmaps.append(heatmap)

        plt.axis('off')
        plt.imshow(utils.stitch_images(heatmaps))
        plt.title('Saliency map')
        plt.savefig(savefig_path)

        print('debug')


if __name__ == '__main__':
    # todo: this class should be tested
    # Example
    nv = NetworkVisualizer(weights_name_dir="Training/TrainingPredicatesCNN/Wed_Jun_14_20:25:16_2017",
                           hierarchy_mapping_name="hierarchy_mapping_predicates.p", gpu_num=0)

    nv.visualize_conv_layers()
