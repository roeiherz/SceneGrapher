import numpy
import cv2

__author__ = 'roeih'

SAMPLES_LIST = 'samples'
STANDART_SCALE_WINDOW_SIZE = 64.0


class BOX:
    def __init__(self):
        pass

    X1 = 0
    Y1 = 1
    X2 = 2
    Y2 = 3


def is_duplicated(boxes, candidate_box):
    """
    Calculates if candidate_box are close enough to the other boxes
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is 'x2', boxes[3] is 'y2'.
    :return: boolean
    """
    iou_val = iou(boxes, candidate_box)
    iou_trsh = iou_val > 0.25
    overlap_val = maximum_overlap(boxes, candidate_box)
    overlap_trsh = overlap_val > 0.75
    cross_val = cross_intersection(boxes, candidate_box)
    return iou_trsh | overlap_trsh | cross_val


def iou(boxes, candidate_box):
    """
    Calculates the intersection over union of a given box and an array of boxes.
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is 'x2', boxes[3] is 'y2'.
    :return: IOU's vector
    """

    intersection_value = numpy.array(intersection(boxes, candidate_box), dtype='float')

    union = box_area(boxes) + box_area(candidate_box) - intersection_value

    if min(union) > 0:
        iou_value = intersection_value / union
    else:
        print('Notice that union here is non positive - which is weird -> better check!')
        iou_value = None

    return iou_value


def find_closest_box(boxes, candidate_box):
    """
    Finds the closest box to a candidate box.
    :param boxes: A list of boxes (dictionary with keys x1, x2, y1, y2).
    :param candidate_box: The candidate box to compare with (dictionary with keys x1, x2, y1, y2).
    :return: The best box, and its IOU with the candidate box.
    :rtype: dict, float, int
    """
    best_iou = 0
    best_box = None
    best_index = -1

    if boxes is None:
        return best_box, best_iou, best_index

    boxes = reshape_vector(boxes)

    if len(boxes) > 0:
        ious = iou(boxes, candidate_box)
        best_index = numpy.argmax(ious)
        best_iou = ious[best_index]
        best_box = boxes[best_index, :]

    return best_box, best_iou, best_index


def box_area(boxes):
    """
    Calculates a box or boxes area.
    :param boxes: A list of boxes or a box (dictionary with keys x1, x2, y1, y2).
    :rtype: dict, float, int
    """
    boxes = reshape_vector(boxes)
    area_value = (boxes[:, BOX.X2] - boxes[:, BOX.X1]) * (boxes[:, BOX.Y2] - boxes[:, BOX.Y1])

    return area_value


def boxes_overlap(boxes, candidate_box):
    """
    Calculates the boxes overlap of a given box and an array of boxes (between one the candidate and boxes).
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is 'x2', boxes[3] is 'y2'.
    :return: overlap ratios vector between one the candidate and boxes
    """

    boxes = reshape_vector(boxes)

    intersection_value = intersection(boxes, candidate_box)
    boxes_area = box_area(boxes)

    indices = boxes_area > 0
    minimum_area_divide = boxes_area * indices + (1 - indices)
    overlap_value = (intersection_value * indices) / numpy.cast['float32'](minimum_area_divide)

    return overlap_value


def maximum_overlap(boxes, candidate_box):
    """
    Calculates the maximum overlap of a given box and an array of boxes.
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is 'x2', boxes[3] is 'y2'.
    :return: maximum overlap ratios vector
    """

    boxes = reshape_vector(boxes)

    intersection_value = intersection(boxes, candidate_box)

    candidate_area = box_area(candidate_box)
    boxes_area = box_area(boxes)
    minimum_area = numpy.minimum(boxes_area, candidate_area)

    indices = minimum_area > 0
    minimum_area_divide = minimum_area * indices + (1 - indices)
    max_overlap_value = (intersection_value * indices) / numpy.cast['float32'](minimum_area_divide)

    return max_overlap_value


def intersection(boxes, candidate_box):
    """
    Calculates the intersection  of a given box and an array of boxes.
    :param candidate_box: Either a box array or an index.
    :param boxes: An array where boxes[0] is 'x1', boxes[1] is 'y1', boxes[2] is width, boxes[3] is height.
    :return: intersection vector
    """

    boxes = reshape_vector(boxes)

    intersection_value = \
        numpy.maximum(0, numpy.minimum(boxes[:, BOX.X2], candidate_box[BOX.X2]) -
                      numpy.maximum(boxes[:, BOX.X1], candidate_box[BOX.X1])) * \
        numpy.maximum(0, numpy.minimum(boxes[:, BOX.Y2], candidate_box[BOX.Y2]) -
                      numpy.maximum(boxes[:, BOX.Y1], candidate_box[BOX.Y1]))

    return intersection_value


def cross_intersection(boxes, candidate_box, intersection_value=-1):
    """
    :param boxes: the boxes
    :param candidate_box: the candidate box
    :param intersection_value:
    :return: Return boolean array whether each box from boxes is crossed with candidate box
    """
    boxes = reshape_vector(boxes)
    # all corners are outside of intersection (and intersection exists)
    if not isinstance(intersection_value, numpy.ndarray):
        intersection_value = intersection(boxes, candidate_box)
    output = numpy.zeros(len(boxes))
    output += -(numpy.sign(boxes[:, BOX.X1] - candidate_box[BOX.X1]))
    output += numpy.sign(boxes[:, BOX.X2] - candidate_box[BOX.X2])
    output += numpy.sign(boxes[:, BOX.Y1] - candidate_box[BOX.Y1])
    output += -(numpy.sign(boxes[:, BOX.Y2] - candidate_box[BOX.Y2]))
    return (numpy.abs(output) == 4) * (intersection_value > 0)


def reshape_vector(ndarr):
    """
    :param ndarr: take list and transform it to a ndarray with reshape
    :return: numpy array of numpy array
    """
    if not isinstance(ndarr, numpy.ndarray):
        # If ndarr is not a ndarray raise exception
        msg = 'This is not a ndarray type: type{}'.format(type(ndarr))
        print(msg)
        raise TypeError(msg)

    if len(ndarr.shape) == 1:
        if len(ndarr) == 0:
            print('ndarray is empty, will not reshape')
            return ndarr
        ndarr_mat = ndarr.copy()
        ndarr_mat.resize(1, ndarr.size)
        return ndarr_mat
    return ndarr


# todo this function was not tested on list of boxes only one box in boxes_a and boxes_b
def find_union_box(boxes_a, boxes_b):
    """
    This function get two list of boxes and returns the union box between them
    :param boxes_a: a numpy array of boxes
    :param boxes_b: a numpy array of boxes
    :return: a numpy array of boxes (which are united)
    """

    # Convert to numpy array
    # boxes_a = numpy.array(boxes_a)
    # boxes_b = numpy.array(boxes_b)

    boxes_a = reshape_vector(boxes_a)
    boxes_b = reshape_vector(boxes_b)
    # Get [[x1,x2]...]
    boxes_a_xs = boxes_a[:, BOX.X1::BOX.X2]
    # Get [[y1,y2]...]
    boxes_a_ys = boxes_a[:, BOX.Y1::BOX.Y2 - 1]
    # Get [[x1,x2]...]
    boxes_b_xs = boxes_b[:, BOX.X1::BOX.X2]
    # Get [[y1,y2]...]
    boxes_b_ys = boxes_b[:, BOX.Y1::BOX.Y2 - 1]

    # Join the all xs and all the ys from the 2 boxes
    all_xs = numpy.concatenate((boxes_a_xs, boxes_b_xs), axis=1)
    all_ys = numpy.concatenate((boxes_a_ys, boxes_b_ys), axis=1)

    # Sort xs and ys
    xs_min = numpy.min(all_xs)
    xs_max = numpy.max(all_xs)
    ys_min = numpy.min(all_ys)
    ys_max = numpy.max(all_ys)

    return numpy.array([xs_min, ys_min, xs_max, ys_max])
