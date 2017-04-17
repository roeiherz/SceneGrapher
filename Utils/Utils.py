import  numpy as np
from numpy.core.umath_tests import inner1d

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
