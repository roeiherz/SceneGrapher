import numpy

__author__ = 'roeih'


class Detections(numpy.ndarray):
    Id = 'id'
    SubjectBox = 'subject_box'
    SubjectId = 'subject_id'
    ObjectBox = 'object_box'
    ObjectId = 'object_id'
    Predicate = 'predicate'
    UnionFeature = 'union_feature'
    UnionBox = 'union_box'
    SubjectClassifications = 'subject_classifications'
    PredictSubjectClassifications = 'predict_subject_classifications'
    ObjectClassifications = 'object_classifications'
    PredictObjectClassifications = 'predict_object_classifications'
    SubjectConfidence = 'subject_confidence'
    ObjectConfidence = 'object_confidence'
    Url = 'url'

    dtype_preset = [(Id, 'int32'),
                    (SubjectBox, '4int32'),
                    (SubjectId, 'int32'),
                    (SubjectClassifications, dict),
                    (SubjectConfidence, numpy.ndarray),
                    (PredictSubjectClassifications, dict),
                    (Predicate, dict),
                    (ObjectBox, '4int32'),
                    (ObjectId, 'int32'),
                    (ObjectClassifications, dict),
                    (ObjectConfidence, numpy.ndarray),
                    (PredictObjectClassifications, dict),
                    (UnionBox, numpy.ndarray),
                    (UnionFeature, numpy.ndarray),
                    (Url, dict)]

    def __new__(cls, *args, **kwargs):
        return super(Detections, cls).__new__(cls, dtype=Detections.dtype_preset, *args, **kwargs)

    def __str__(self):
        if len(self.shape) == 0:
            size = 0
        else:
            size = self.shape[0]
        return '{0} Detections'.format(size)

    def __repr__(self):
        return self.__str__()
