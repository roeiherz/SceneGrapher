from __future__ import print_function
import cPickle
import os
import random
import numpy as np
from DesignPatterns.Detections import Detections
import collections
from DesignPatterns.DimensionalityReduction import tSNEReducer, PCAReducer
from DesignPatterns.TwoDVisualization import TwoDVisualizer
from Utils import get_detections

__author__ = 'roeih'

NOF_SAMPLES = 1000

if __name__ == '__main__':
    print('tSNE_Test')
    data_directory = ''
    detections = get_detections(detections_file_name="predicated_mini_fixed_detections_url.p")

    # Create a numpy array of indices of the data
    # indices = np.arange(len(detections))
    # unique, counts = np.unique(detections[Detections.Predicate], return_counts=True)
    # occurrences1 = dict(zip(unique, counts))

    # Get the counts of the labels
    counts = collections.Counter(detections[Detections.Predicate])
    # Get the top K occurrences labels
    occurrences = [key for key, val in counts.most_common(4)]
    # Get the lowest K occurrences labels
    # occurrences = sorted(counts)[-4:]

    print("Get the following labels: {}".format(occurrences))

    # Find the top K indices labels
    indices = np.where(np.in1d(list(detections[Detections.Predicate]), occurrences) == True)[0]
    # Shuffle the indices of the data
    # random.shuffle(indices)
    # Get random indices
    # rand_indices = indices[:NOF_SAMPLES]
    rand_indices = indices

    # data is [nof_sample, 2048]
    data = np.concatenate(detections[Detections.UnionFeature][rand_indices])
    # labels is [nof_sample, 1]
    labels = np.array(detections[Detections.Predicate][rand_indices])

    print("Data shape: {0} and labels shape: {1}".format(data.shape, labels.shape))

    visualizer = TwoDVisualizer()
    reducer_tsne = tSNEReducer()
    reducer_pca = PCAReducer()
    reduced_data, reducer_fitted = visualizer.reduce(reducer_tsne, data, labels, goal_dimension=2)
    figure = visualizer.visualize_2d_data(reduced_data, labels, data_directory)

    cPickle.dump(reduced_data, open(os.path.join(data_directory, 'reduced_Data.p'), 'wb+'))
    print('finished')
