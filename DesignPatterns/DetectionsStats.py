import numpy as np
import sys

sys.path.append("..")
from features_extraction.Utils.Boxes import find_union_box
from features_extraction.Utils.Utils import get_mask_from_object

__author__ = 'roeih'


# todo: should implements has equal()
class DetectionsStats(np.ndarray):
    Id = 'id'
    SubjectBox = 'subject_box'
    SubjectId = 'subject_id'
    ObjectBox = 'object_box'
    ObjectId = 'object_id'
    Predicate = 'predicate'
    PredictPredicate = 'predict_predicate'
    UnionBox = 'union_box'
    SubjectClassifications = 'subject_classifications'
    PredictSubjectClassifications = 'predict_subject_classifications'
    ObjectClassifications = 'object_classifications'
    PredictObjectClassifications = 'predict_object_classifications'
    SubjectConfidence = 'subject_confidence'
    ObjectConfidence = 'object_confidence'
    Url = 'url'
    TopKIndex = "top_k_index"
    IsGT = "is_gt"
    RelationConfidence = "relation_confidence"

    dtype_preset = [(Id, 'int32'),
                    (SubjectBox, '4int32'),
                    (SubjectId, 'int32'),
                    (SubjectClassifications, dict),
                    (SubjectConfidence, np.ndarray),
                    (PredictSubjectClassifications, dict),
                    (Predicate, dict),
                    (PredictPredicate, dict),
                    (ObjectBox, '4int32'),
                    (ObjectId, 'int32'),
                    (ObjectClassifications, dict),
                    (ObjectConfidence, np.ndarray),
                    (PredictObjectClassifications, dict),
                    (UnionBox, np.ndarray),
                    (Url, dict),
                    (TopKIndex, 'int32'),
                    (IsGT, bool),
                    (RelationConfidence, 'float32')]

    def __new__(cls, *args, **kwargs):
        return super(DetectionsStats, cls).__new__(cls, dtype=DetectionsStats.dtype_preset, *args, **kwargs)

    def __str__(self):
        if len(self.shape) == 0:
            size = 0
        else:
            size = self.shape[0]
        return '{0} DetectionsStats'.format(size)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def set_detections(filtered_id, vg_subject, vg_object, predicted_subject, predicted_object, predicted_predicate,
                       predicate_gt,
                       top_k_index, is_gt, relation_confidence, url, detection_stats):
        """
        This function set the following inputs for a DetectionsStats object
        :param filtered_id: A filtered id
        :param predicate_gt: A predicate gt class9
        :param vg_subject: Subject which is a Visual Genome class
        :param vg_object: Object which is a Visual Genome class
        :param predicted_subject: A string which is a prediction to Subject  
        :param predicted_object: A string which is a prediction to Object 
        :param predicted_predicate: A string which is a prediction to Predicate 
        :param top_k_index: the index which the relation <i,k,j> is predicted between the top k relations prediction
        :param is_gt: A boolean to decide if is a GT or not
        :param relation_confidence: A float which is the confidence of the predicted relation
        :param url: A url entity
        :param detection_stats: a Detections index which we will be set
        :return: 
        """

        # Update Relation Id
        detection_stats[DetectionsStats.Id] = filtered_id
        # Update Subject Id
        detection_stats[DetectionsStats.SubjectId] = vg_subject.id
        # Get the mask: a dict with {x1,x2,y1,y2}
        mask_subject = get_mask_from_object(vg_subject)
        # Saves as a box
        subject_box = np.array([mask_subject['x1'], mask_subject['y1'], mask_subject['x2'], mask_subject['y2']])
        # Update Subject Box
        detection_stats[DetectionsStats.SubjectBox] = subject_box
        # Update Object Id
        detection_stats[DetectionsStats.ObjectId] = vg_object.id
        # Get the mask: a dict with {x1,x2,y1,y2}
        mask_object = get_mask_from_object(vg_object)
        # Saves as a box
        object_box = np.array([mask_object['x1'], mask_object['y1'], mask_object['x2'], mask_object['y2']])
        # Update Object box
        detection_stats[DetectionsStats.ObjectBox] = object_box
        # Update Subject Classification
        detection_stats[DetectionsStats.SubjectClassifications] = vg_subject.names[0]
        # Update Object Classification
        detection_stats[DetectionsStats.ObjectClassifications] = vg_object.names[0]
        # Update Url
        detection_stats[DetectionsStats.Url] = url
        # Update Predicate
        detection_stats[DetectionsStats.Predicate] = predicate_gt
        # Update UnionBox
        detection_stats[DetectionsStats.UnionBox] = find_union_box(subject_box, object_box)
        # Update Predicted Subject
        detection_stats[DetectionsStats.PredictSubjectClassifications] = predicted_subject
        # Update Predicted Object
        detection_stats[DetectionsStats.PredictObjectClassifications] = predicted_object
        # Update Predicted Predicate
        detection_stats[DetectionsStats.PredictPredicate] = predicted_predicate
        # Update Top-K index
        detection_stats[DetectionsStats.TopKIndex] = top_k_index
        # Update a flag Is it a GT
        detection_stats[DetectionsStats.IsGT] = is_gt
        # Update the relation confidence
        detection_stats[DetectionsStats.RelationConfidence] = relation_confidence

    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #
    #         # Shape is not as the same size
    #         if len(self.shape[0]) != len(other.shape[0]):
    #             return False
    #
    #         # Detection is the same
    #         for i in self.shape[0]:
    #             if not self.detection_equals(self[i], other[i]):
    #                 return False
    #             return True
    #     else:
    #         return False
    #
    # @staticmethod
    # def detection_equals(one, other):
    #     """
    #     This function check if 2 detections are equals
    #     :param one:
    #     :param other:
    #     :return:
    #     """
    #     pass
