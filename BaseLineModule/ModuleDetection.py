from DesignPatterns.DetectionsStats import DetectionsStats
import os

from FilesManager.FilesManager import FilesManager


class ModuleDetection():
    """
    Holds and maintain Image detection (used to evaluate the module)
    """
    def __init__(self, entity, module):
        """
        Constructor
        :param entity: visual genome entity
        :param module: the module to evaluate
        """
        # store module
        self.module = module
        # store the entity
        self.entity = entity
        # detections list
        self.detections_params = []

    def add_detection(self, global_subject_id, global_object_id, pred_subject, pred_object, pred_predicate, top_k_index, confidence):
        """
        Add new detection to DB
        :param global_subject_id: global object id from visual genome
        :param global_object_id: global object id from visual genome
        :param pred_subject: module prediction id of subject
        :param pred_object: module prediction id of object
        :param pred_predicate: module prediction id of predicate
        :param top_k_index: detection index ordered by confidence
        :param confidence: the actual confidence
        :return:
        """
        # finc visual genome subject and object
        vg_subject = None
        vg_object = None
        # find subject and object
        for object in self.entity.objects:
            if object.id == global_subject_id:
                vg_subject = object
            if object.id == global_object_id:
                vg_object = object

        # find if exist in gt and gt_predicate
        is_gt = 0
        true_predicate = "neg"
        filtered_id = -1
        for relation in self.entity.relationships:
            if relation.subject.id == global_subject_id and relation.object.id == global_object_id:
                if relation.predicate != "neg":
                    is_gt = 1
                true_predicate = relation.predicate
                filtered_id = relation.filtered_id
                break

        # predicated ids to names
        pred_subject_name = self.module.reverse_object_ids[pred_subject]
        pred_object_name = self.module.reverse_object_ids[pred_object]
        pred_predicate_name = self.module.reverse_predicate_ids[pred_predicate]

        # store detection params to be converted to numpy object later
        detection = {"vg_subject" : vg_subject, "vg_object" : vg_object, "true_predicate" : true_predicate,
                     "pred_subject" : pred_subject_name, "pred_object" : pred_object_name, "pred_predicate" : pred_predicate_name,
                     "is_gt" : is_gt, "top_k_index" : top_k_index, "confidence" : confidence, "filtered_id" : filtered_id}
        self.detections_params.append(detection)

    def get_stat(self):
        stat = DetectionsStats(len(self.detections_params))
        for i in range(len(self.detections_params)):
            params =  self.detections_params[i]
            stat.set_detections(filtered_id=params["filtered_id"], vg_subject=params["vg_subject"], vg_object=params["vg_object"], predicted_subject=params["pred_subject"], predicted_object=params["pred_object"], predicted_predicate=params["pred_predicate"], predicate_gt=params["true_predicate"], top_k_index=params["top_k_index"], is_gt=params["is_gt"], relation_confidence=params["confidence"], url=self.entity.image.url, detection_stats=stat[i])

        return stat

    def save_stat(self, score=0):
        filesManager = FilesManager()
        stat = self.get_stat()
        if filesManager.file_exist("scene_graph_base_module.eval.stat"):
            stat_lst = filesManager.load_file("scene_graph_base_module.eval.stat")
        else:
            stat_lst = []
        stat_lst.append((stat, self.entity, score))

        filesManager.save_file("scene_graph_base_module.eval.stat", stat_lst)
