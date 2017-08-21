import os
import sys
import numpy as np
from numpy.core.umath_tests import inner1d
from FilesManager import FilesManager
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

FILE_EXISTS_ERROR = (17, 'File exists')


def cosine_distance(u, v):
    """
    Get array / matrix u and v and calculate the cosine distance for each row separately
    :param u: array represents a vector or matrix where each row represent vector
    :param v: array represents a vector or matrix where each row represent vector
    :return: returns scalar reprensts the cosine distance or array where each element represent the cosine distance of each row.
    """
    norm_u = u / np.sqrt((u * u).sum(axis=1)).reshape(-1, 1)
    norm_v = v / np.sqrt((v * v).sum(axis=1)).reshape(-1, 1)
    cosine_dist_vec = inner1d(norm_u, norm_v)
    return cosine_dist_vec


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem assignment1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Implementation Matrix

        # Get the maximum each row from data
        max_rows = np.max(x, axis=1)
        # Reshape the max rows to a matrix [:,assignment1]
        reshape_max_rows = max_rows.reshape((max_rows.shape[0]), 1)
        # Normalize the matrix by subtract the max from each row per row
        norm_data = x - reshape_max_rows
        # Power mat by exponent
        exp_mat = np.exp(norm_data)
        # Sum each col per exp_mat
        exp_mat_rows_sum = np.sum(exp_mat, axis=1)
        # The new SoftMax mat is exp_mat normalized by the rows_sum
        x = exp_mat / exp_mat_rows_sum.reshape(-1, 1)

    else:
        # Implementation a row Vector

        # Get the maximum each row from data
        max_rows = np.max(x)
        # Normalize the matrix by subtract the max from each row per row
        norm_data = x - max_rows
        # Power mat by exponent
        exp_mat = np.exp(norm_data)
        # Sum each col per exp_mat
        exp_mat_rows_sum = np.sum(exp_mat, axis=0)
        # The new SoftMax mat is exp_mat normalized by the rows_sum
        x = exp_mat / exp_mat_rows_sum
        return x

    assert x.shape == orig_shape
    return x


def softmax_multi_dim(x):
    """Compute softmax values for a multidim array"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def create_folder(path):
    """
    Checks if the path exists, if not creates it.
    :param path: A valid path that might not exist
    :return: An indication if the folder was created
    """
    folder_missing = not os.path.exists(path)

    if folder_missing:
        # Using makedirs since the path hierarchy might not fully exist.
        try:
            os.makedirs(path)
        except OSError as e:
            if (e.errno, e.strerror) == FILE_EXISTS_ERROR:
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


def get_detections():
    """
    This function gets the whole filtered detections data (with no split between the  modules)
    :return: detections
    """
    filesmanager = FilesManager.FilesManager()
    detections = filesmanager.load_file("scene_graph_base_module.visual_module.detections")
    return detections


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    sp_sparse = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    adj_mat = np.zeros(sp_sparse.shape)
    adj_mat[(sp_sparse.row, sp_sparse.col)] = sp_sparse.data
    return adj_mat


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized