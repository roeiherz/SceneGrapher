import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import sin, cos, pi

from Utils import create_folder

__author__ = 'roeih'


class TwoDVisualizer(object):
    @staticmethod
    def reduce(reducer, data, labels, goal_dimension=2):

        # Check the nof samples of the data is more than 1
        if data.shape[0] == 1 or labels.shape[0] == 1:
            print("Error the number of samples should be more than 1!")
            exit()

        data_reduced, reducer_fitted = reducer.reduce(data, labels, goal_dimension)
        return data_reduced, reducer_fitted

    @staticmethod
    def rotate_2D_points(points, rotation_angle):
        """
        Given an array of points (2D!!), this function will rotate the points in a given angle anti - clockwise
        :param points: array of 2D points
        :type points: numpy.ndarray
        :param rotation_angle:
        :type rotation_angle: float
        :return:
        """
        rotation_radians = rotation_angle * pi / 180
        rotation_matrix = np.array(
            [[cos(rotation_radians), -sin(rotation_radians)], [sin(rotation_radians), cos(rotation_radians)]])
        point_transposed = np.transpose(points)
        rotated_points = np.transpose(np.dot(rotation_matrix, point_transposed))
        return rotated_points

    @staticmethod
    def plotting_and_saving_2d_data(data, color='red', marker='x', figure=None, fig_filename=None, classname=None,
                                    errorbar_y=None):
        if figure is None:
            figure = plt.figure()
            ax = figure.add_subplot(111)
        else:
            ax = figure.add_subplot(111)

        I1 = ax.scatter(data[:, 0], data[:, 1], c=color, marker=marker, label=classname, s=150, alpha=0.3)
        plt.errorbar(data[:, 0], data[:, 1], yerr=errorbar_y, ecolor=color, color=color)

        if fig_filename:
            figure.set_size_inches(20, 15)
            figure.savefig(fig_filename)
        return figure, I1

    @staticmethod
    def visualize_2d_data(data, labels=None, figure_filename=None, color_dict=None, x_label=None, y_label=None,
                          figure=None, error_bar_y=None, color='r'):
        """
        This function receives 2D data with its labels and plots the data using different (random) color & marker for
        each class.
        :param color:
        :param error_bar_y:
        :param data: 2D array (data)
        :type data: numpy.ndarray
        :param labels: array on integers which labels the data
        :type labels: numpy.ndarray
        :param figure_filename: if figure_filename is given, the figure will be saved with respect to the given path and filename.
        :type figure_filename: str
        :param color_dict: a dictionary which maps labels to color. if missing a random color will be chosen for each class
        :type color_dict: dict
        :param x_label: label for x axis
        :type x_label: str
        :param y_label:label for y axis
        :type y_label: str
        :param figure: if figure is given - the pot will be added upon that figure else, a new figure will be returned
        :type figure: plt.figure
        :return: returns the figure
        :rtype: plt.figure
        """

        if labels is not None:
            if data.shape[0] != labels.shape[0]:
                print('Data and Labels shapes don"t match!! please check!')
                return

        marker = itertools.cycle(('o'))

        # Unique labels
        unique_labels = np.unique(labels)

        print("Different Labels: {}".format(len(unique_labels)))

        for i in range(len(unique_labels)):

            # Get the label
            label_name = unique_labels[i]

            # Get the data
            class_features = data[np.where(labels == label_name)[0]]

            if color_dict is None:
                color = np.random.rand(3, 1)
            else:
                if not color:
                    color = color_dict[unique_labels[i]]

            figure, I1 = TwoDVisualizer.plotting_and_saving_2d_data(class_features, color=color, marker=marker.next(),
                                                                    figure=figure, classname=label_name,
                                                                    errorbar_y=error_bar_y)
        ax = figure.get_axes()
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)

        # axes labels
        if x_label:
            plt.xlabel(x_label, alpha=0.5)
        if y_label:
            plt.ylabel(y_label, alpha=0.5)

        figure.set_size_inches(20, 16)
        if figure_filename:
            if os.path.isfile(figure_filename):
                print('File already exists - will be overwritten')
            create_folder(os.path.dirname(figure_filename))
            figure.savefig(figure_filename)
        return figure
