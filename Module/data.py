import os
import os.path
import numpy as np
import cPickle
import operator


def prepare_data():
    """
    Basic implementation to prepare data for language module training.
    TBD: improve
    :return: filtered_data
    """

    # Check if filtered data already exist
    if os.path.isfile("final_module_data.p"):
        module_data_file = open("filtered_data.p", "rb")
        module_data = cPickle.load(module_data_file)
        module_data_file.close()
    else:
        # load entities and filter it according to most popular classes and predicates
        module_data_file = open("filtered_module_data.p", "rb")
        module_data = cPickle.load(module_data_file)
        module_data_file.close()

        object_ids = module_data["object_ids"]
        predicate_ids = module_data["predicate_ids"]
        entities = module_data["module_entities"]

        # split entities to train, validation and test
        nof_entities = len(entities)
        train_entities = entities[:int(0.6 * nof_entities)]
        validation_entities = entities[int(0.6 * nof_entities):int(0.8 * nof_entities)]
        test_entities = entities[int(0.8 * nof_entities):]

        # create data object for train, validation, test
        train_data = convert_entities_to_data(train_entities, object_ids, predicate_ids)
        validation_data = convert_entities_to_data(validation_entities, object_ids, predicate_ids)
        test_data = convert_entities_to_data(test_entities, object_ids, predicate_ids)

        # save data
        module_data = {"train": train_data, "validation": validation_data, "test": test_data, "object_ids": object_ids,
                       "predicate_ids": predicate_ids}
        filtered_data_file = open("final_module_data.p", "wb")
        cPickle.dump(module_data, filtered_data_file, 0)
        filtered_data_file.close()

    module_data["train"] = Data(module_data["train"])
    module_data["validation"] = Data(module_data["validation"])
    module_data["test"] = Data(module_data["test"])

    return module_data


def convert_entities_to_data(entities, object_ids, predicate_ids):
    """
    Creating object of data given list of entities
    :param entities: list of entities from visual genome
    :param object_ids: dictionary convert object to object_id
    :param predicate_ids: dictionary convert predicate to predicate_id
    :return: instance of data object
    """
    worda = []
    wordb = []
    predicate = []
    data_predicate_ids = []
    data_relation_ids = []
    data_subject_ids = []
    data_object_ids = []
    instances = {}
    for entity in entities:
        for R in entity.relationships:
            # filter relationships
            if not object_ids.has_key(R.subject.names[0]):
                continue
            if not object_ids.has_key(R.object.names[0]):
                continue
            if not predicate_ids.has_key(R.predicate):
                continue

            # add subject
            worda.append(R.subject.names[0])
            data_subject_ids.append(object_ids[R.subject.names[0]])
            # add object
            wordb.append(R.object.names[0])
            data_object_ids.append(object_ids[R.object.names[0]])
            # add predicate
            predicate.append(R.predicate)
            data_predicate_ids.append(predicate_ids[R.predicate])
            # add relation id
            data_relation_ids.append(R.filtered_id)
            # maintain number of instances per relationship
            r_str = R.subject.names[0] + "-" + R.predicate + "-" + R.object.names[0]
            if instances.has_key(r_str):
                instances[r_str] += 1
            else:
                instances[r_str] = 1

    worda = np.asarray(worda)
    wordb = np.asarray(wordb)
    predicate = np.asarray(predicate)
    data_predicate_ids = np.asarray(data_predicate_ids)
    data_relation_ids = np.asarray(data_relation_ids)
    data_subject_ids = np.asarray(data_subject_ids)
    data_object_ids = np.asarray(data_object_ids)

    filtered_data = {"subjects": worda, "objects": wordb, "predicates": predicate, "predicates_ids": data_predicate_ids,
                     "subject_ids": data_subject_ids, "object_ids": data_object_ids, "relation_ids": data_relation_ids,
                     "instances": instances}

    return filtered_data


class Data(object):
    """
    Class grouped the list of relationship,, used by language module training.
    """

    def __init__(self, data=None):
        if data != None:
            # save original input
            self.orig_data = data

            # create class fields
            self.worda = data["subjects"]
            self.wordb = data["objects"]
            self.predicate = data["predicates"]
            self.predicate_ids = data["predicates_ids"]
            self.relation_ids = data["relation_ids"]
            self.subject_ids = data["subject_ids"]
            self.subject_ids = np.asarray([0] * len(self.worda))
            self.object_ids = data["object_ids"]

            self.instances = np.zeros(len(self.predicate))
            for i in range(len(self.predicate)):
                self.instances[i] = data["instances"][self.worda[i] + "-" + self.predicate[i] + "-" + self.wordb[i]]

    def get_subset(self, indices):
        """
        Return subset of the data
        :param indices:  the required data indecies
        :return: subset of the data
        """
        sub_data = Data()
        sub_data.worda = self.worda[indices]
        sub_data.wordb = self.wordb[indices]
        sub_data.predicate = self.predicate[indices]
        sub_data.predicate_ids = self.predicate_ids[indices]
        sub_data.relation_ids = self.relation_ids[indices]
        sub_data.subject_ids = self.subject_ids[indices]
        sub_data.object_ids = self.object_ids[indices]
        sub_data.instances = self.instances[indices]

        return sub_data

    def split(self, percent):
        part1 = Data()
        part2 = Data()

        part1_max = int(len(self.worda) * percent)
        part1.worda = self.worda[0:part1_max]
        part1.wordb = self.wordb[0:part1_max]
        part1.predicate = self.predicate[0:part1_max]
        part1.predicate_ids = self.predicate_ids[0:part1_max]
        part1.relation_ids = self.relation_ids[0:part1_max]
        part1.subject_ids = self.subject_ids[0:part1_max]
        part1.object_ids = self.object_ids[0:part1_max]
        part1.instances = self.instances[0:part1_max]

        part2.worda = self.worda[part1_max + 1:]
        part2.wordb = self.wordb[part1_max + 1:]
        part2.predicate = self.predicate[part1_max + 1:]
        part2.predicate_ids = self.predicate_ids[part1_max + 1:]
        part2.relation_ids_ids = self.relation_ids[part1_max + 1:]
        part2.subject_ids = self.subject_ids[part1_max + 1:]
        part2.object_ids = self.object_ids[part1_max + 1:]
        part2.instances = self.instances[part1_max + 1]

        return part1, part2


if __name__ == "__main__":
    prepare_data()
