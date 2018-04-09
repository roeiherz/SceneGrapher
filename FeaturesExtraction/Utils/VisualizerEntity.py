import traceback
import scipy
import cv2
import numpy
import os
from FeaturesExtraction.Utils.Boxes import BOX
from FeaturesExtraction.Utils.Utils import get_img, get_mask_from_object
from FeaturesExtraction.Utils.data import draw_graph

__author__ = 'roeih'

FONT_FACE = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 1
FONT_THICKNESS = 2
LINE_THICKNESS = -1
FONT_COLOR = (0, 0, 0)
GREEN_COLOR = (0, 255, 0)
FILL_COLOR = (255, 255, 0)
PADDING = 1
SCALE_FACTOR_DEF = 500

color_vec = [(255, 255, 255), (0, 125, 0), (0, 0, 125), (255, 255, 0), (255, 0, 255), (0, 255, 255),
             (0, 125, 125), (125, 0, 125), (125, 125, 0), (255, 125, 125), (125, 255, 125), (125, 125, 255),
             (255, 255, 125), (255, 125, 255), (125, 255, 255), (0, 125, 255), (125, 0, 255), (125, 255, 0),
             (0, 255, 125), (255, 125, 0), (255, 0, 125), (60, 0, 125), (60, 125, 0), (125, 60, 0),
             (0, 125, 125), (125, 0, 125), (125, 125, 0)]


def rect_color(rect_id):
    return color_vec[rect_id % len(color_vec)]


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
            if (e.errno, e.strerror) == (17, 'File exists'):
                print(e)
            else:
                raise

        print('Created folder {0}'.format(path))

    return folder_missing


class CvColor:
    def __init__(self):
        pass

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    ORANGE = (0, 128, 255)
    GRAY = (160, 160, 160)
    PURPLE = (102, 0, 102)
    YELLOW = (0, 255, 255)


class VisualizerEntity(object):
    def __init__(self, entity, path):
        self.canvases = {}
        self.entity = entity
        self.url = entity.image.url
        self.id = entity.image.id
        # Get image
        self.image = get_img(self.url, download=True)
        create_folder(path)
        self.create_canvas(self.id)
        self.folder_path = path
        self.draw_num = 0
        self.color_dict = {0: cv2.COLORMAP_AUTUMN, 1: cv2.COLORMAP_WINTER, 2: cv2.COLORMAP_JET, 3: cv2.COLORMAP_WINTER,
                           4: cv2.COLORMAP_SPRING, 5: cv2.COLORMAP_SUMMER, 6: cv2.COLORMAP_OCEAN}

    def create_canvas(self, name):
        if name not in self.canvases:
            canvas = self.image.copy()
            self.canvases[name] = canvas
            return name
        else:
            return None

    def get_color_map(self, image, color=cv2.COLORMAP_JET):
        return cv2.applyColorMap(cv2.equalizeHist(image), color)

    def draw_attention(self, confidences):
        original_detection_centers = {}
        i = 0
        for object in self.entity.objects:
            try:
                confidences_per_object = confidences[:, i]
                heat_map = numpy.zeros(shape=[self.image.shape[0], self.image.shape[1], 1], dtype=numpy.float64)
                original_detection_centers = self.attention_per_neighb(original_detection_centers, confidences_per_object, heat_map)

                cv2.normalize(heat_map, heat_map, 0, 255, cv2.NORM_MINMAX)
                heat_map = cv2.convertScaleAbs(heat_map)
                heat_map_float = heat_map / (heat_map.max() / 16.)
                heat_map_float = numpy.power(heat_map_float, 2)
                heat_map_int = cv2.normalize(heat_map_float, None, 0, 255, cv2.NORM_MINMAX)
                heat_map_int = cv2.convertScaleAbs(heat_map_int)
                heat_map = heat_map_int

                color_map = self.get_color_map(heat_map)
                color_map[heat_map == 0] = self.image[heat_map == 0]
                blend = cv2.addWeighted(color_map, 0.6, self.image, 0.4, 0)

                # path_save = os.path.join(self.folder_path, "objects_img_{0}_att_{1}.jpg".format(self.entity.image.id, object))
                path_save = os.path.join("objects_img_{0}_att_{1}.jpg".format(self.entity.image.id, object))
                cv2.imwrite(path_save, blend)
                print("Objects image have been saved in {} \n".format(path_save))
                i += 1

            except Exception as e:
                print("Error: {}".format(str(e)))
                traceback.print_exc()

    def attention_per_neighb(self, original_detection_centers, confidences_per_object, heat_map, scale=2):

        ind = 0
        # output = self.image.copy()
        for object in self.entity.objects:
            try:
                # overlay = self.image.copy()
                confidence = confidences_per_object[ind]
                # Get the mask: a dict with {x1,x2,y1,y2}
                mask_object = get_mask_from_object(object)
                # Saves as a box
                object_box = numpy.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
                width = object_box[BOX.X2] - object_box[BOX.X1]
                height = object_box[BOX.Y2] - object_box[BOX.Y1]
                object_label_gt = object.names[0]
                original_detection_centers[ind] = (object_box[BOX.X1] + width / 2, object_box[BOX.Y1] + height / 2)
                # self.draw_labeled_box(name=self.id, box=object_box, label=object_label_gt,
                #                       rect_color=rect_color(ind), scale=2, img=overlay)

                # cv2.imwrite("overlay.jpg", overlay)
                # output = cv2.addWeighted(overlay, confidence, output, 1, 0)
                # cv2.imwrite("output.jpg", output)

                gaussian = numpy.zeros(shape=[height, width], dtype=numpy.float64)
                gaussian[height / 2, width / 2] = 1
                gaussian = scipy.ndimage.filters.gaussian_filter(gaussian, (height / 8., width / 8.))
                cv2.normalize(gaussian, gaussian, 0, confidence, cv2.NORM_MINMAX)
                heat_map[object_box[BOX.Y1]:object_box[BOX.Y2], object_box[BOX.X1]:object_box[BOX.X2]] += numpy.expand_dims(gaussian, axis=2)
                # heat_map[object_box[BOX.Y1]:object_box[BOX.Y2], object_box[BOX.X1]:object_box[BOX.X2]] = numpy.expand_dims(gaussian, axis=2)

                # copy_heatmap = numpy.copy(heat_map)
                # cv2.normalize(copy_heatmap, copy_heatmap, 0, 255, cv2.NORM_MINMAX)
                # copy_heatmap = cv2.convertScaleAbs(copy_heatmap)
                # heat_map_float = copy_heatmap / (copy_heatmap.max() / 16.)
                # heat_map_float = numpy.power(heat_map_float, 2)
                # heat_map_int = cv2.normalize(heat_map_float, None, 0, 255, cv2.NORM_MINMAX)
                # heat_map_int = cv2.convertScaleAbs(heat_map_int)
                # copy_heatmap = heat_map_int
                #
                # color_map = self.get_color_map(copy_heatmap, 0)
                # color_map[copy_heatmap == 0] = self.image[copy_heatmap == 0]
                # blend = cv2.addWeighted(color_map, 0.5, self.image, 0.5, 0)

                ind += 1
            except Exception as e:
                print("Error: {}".format(str(e)))
                traceback.print_exc()

        return original_detection_centers

    def draw_objects(self):
        i = 0
        for object in self.entity.objects:
            try:
                # Get the mask: a dict with {x1,x2,y1,y2}
                mask_object = get_mask_from_object(object)
                # Saves as a box
                object_box = numpy.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
                object_label_gt = object.names[0]

                # Draw Object box with their labels
                self.draw_labeled_box(name=self.id, box=object_box, label=object_label_gt,
                                      rect_color=rect_color(i), scale=2)

                i += 1

            except Exception as e:
                print("Error: {}".format(str(e)))
                traceback.print_exc()

        path_save = os.path.join(self.folder_path, "objects_img_{}.jpg".format(self.entity.image.id))
        cv2.imwrite(path_save, self.image)
        print("Objects image have been saved in {} \n".format(path_save))

    def save_image(self, image, name):
        self.draw_num += 1
        img_path = os.path.join(self.folder_path, str(self.draw_num) + '-' + str(name) + '.jpg')
        cv2.imwrite(img_path, image)
        print('Image saved: {}'.format(img_path))

    def save_all(self):
        for name in self.canvases:
            self.save_image(self.canvases[name], name)

    def draw_labeled_box(self, name, box, label=None, rect_color=CvColor.GREEN, scale=None, text_color=None,
                         where="top_left", label2=None, img=None):
        self.create_canvas(name)
        # Drawing the rectangular.
        try:
            if img is None:
                img = self.image

            cv2.rectangle(img, (int(box[BOX.X1]), int(box[BOX.Y1])), (int(box[BOX.X2]), int(box[BOX.Y2])),
                          rect_color, thickness=2 * scale)
        except Exception as e:
            print('error drawing box {} with exception {}'.format(str(box), e))

        if label is not None:
            label = label.encode('utf-8') if isinstance(label, basestring) else str(label)

            label_size = cv2.getTextSize(label, FONT_FACE, scale, FONT_THICKNESS)
            label_pixel_size = label_size[0]
            rect_top_left_pt = (box[BOX.X1] + 10, box[BOX.Y1] + 25)
            if label == "man":
                rect_top_left_pt = (box[BOX.X1] - 30, box[BOX.Y1] + 90)
            label_pixel_height = label_pixel_size[1]
            text_org = tuple(map(lambda x: numpy.float32(sum(x)), zip(rect_top_left_pt, (0, label_pixel_height),
                                                                      (PADDING, 2 * PADDING))))
            if text_color is None:
                text_color = FONT_COLOR
            cv2.putText(img, label, text_org, FONT_FACE, 4, rect_color, 9)
            cv2.putText(img, label, text_org, FONT_FACE, 4, text_color, 3)

    def draw_scene_graph(self):
        """
        This function draws the original graph from scene graph (without "neg" predicate per entity).
        We need to create a new objects and predicates mapping for that.
        :return:
        """
        # Create 3 mapping: objects_ids_to_indices, labels_to_index_objects and index_to_labels_objects
        objects_ids_to_indices = {}
        labels_to_index_objects = {}
        ind = 0
        unique_ind = 0
        for obj in self.entity.objects:

            # Update object_ids to object index mapping
            if obj.id not in objects_ids_to_indices:
                objects_ids_to_indices[obj.id] = ind

            # Update labels to index objects mapping
            if obj.names[0] not in labels_to_index_objects:
                labels_to_index_objects[obj.names[0]] = unique_ind
                unique_ind += 1

            ind += 1
        index_to_labels_objects = {ind: label for label, ind in labels_to_index_objects.iteritems()}

        # Create OBJECTS GT 1D array that has the object_ids inside
        objects_gt_arr = numpy.array([labels_to_index_objects[obj.names[0]] for obj in self.entity.objects])

        # Create 2 mapping: labels_to_index_predicates and index_to_labels_predicates
        labels_to_index_predicates = {}
        ind = 0
        for relation in self.entity.relationships:
            if relation.predicate not in labels_to_index_predicates:
                labels_to_index_predicates[relation.predicate] = ind
                ind += 1

        index_to_labels_predicates = {ind: label for label, ind in labels_to_index_predicates.iteritems()}

        # Create PREDICATES GT 2D array that has the predicates_ids inside
        pred_gt_arr = numpy.zeros((len(self.entity.objects), len(self.entity.objects)), dtype="int8")
        pred_gt_arr.fill(50)
        for relation in self.entity.relationships:
            object_id = relation.object.id
            object_ind = objects_ids_to_indices[object_id]
            subject_id = relation.subject.id
            subject_ind = objects_ids_to_indices[subject_id]
            predicate = relation.predicate
            pred_gt_arr[subject_ind][object_ind] = labels_to_index_predicates[predicate]

        # The file name which will be saved
        file_name = os.path.join(self.folder_path, "scene_graph_{0}".format(self.id))

        # Draw the graph
        draw_graph(only_gt=True, pred=None, pred_gt=pred_gt_arr, obj=None, obj_gt=objects_gt_arr,
                   predicate_ids=index_to_labels_predicates, object_ids=index_to_labels_objects,
                   filename=file_name)
