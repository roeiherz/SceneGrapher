import os
import os.path
import numpy as np
import cPickle
import operator


def prepare_data(max_nof_predicates):
    """
    Basic implementation to prepare data for language module training.
    TBD: improve
    :param max_nof_predicates
    :return: filtered_data
    """

    # Check if filtered data already exist
    if os.path.isfile("filtered_data.p"):
        file = open("filtered_data.p", "rb")
        filtered_data = cPickle.load(file)
        file.close()

        nof_predicates = 150
    else:
        # load entities and filter it according to most popular classes and predicates
        file = open("final_entities.p", "rb")
        entities = cPickle.load(file)
        file.close()

        file = open("final_classes_count.p", "rb")
        classes = cPickle.load(file)
        file.close()

        file = open("final_predicates_count.p", "rb")
        predicates = cPickle.load(file)
        file.close()

        nof_predicates = 150
        sorted_predictes_count = sorted(predicates.items(), key=operator.itemgetter(1), reverse=True)
        sorted_predictes = sorted_predictes_count[:nof_predicates]
        predicates_to_be_used = dict(sorted_predictes)

        nof_classes = 150
        sorted_classes_count = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
        sorted_classes = sorted_classes_count[:nof_classes]
        classes_to_be_used = dict(sorted_classes)

        worda = []
        wordb = []
        predicate = []
        predicate_ids = []
        subject_ids = []
        object_ids = []
        instances = {}
        predicate_dict = {}
        object_dict = {}
        for entity in entities:
            for R in entity.relationships:
                # filter relationships
                if len(predicate_dict) >= max_nof_predicates:
                    break
                if not classes_to_be_used.has_key(R.subject.names[0]):
                    continue
                if not classes_to_be_used.has_key(R.object.names[0]):
                    continue
                if not predicates_to_be_used.has_key(R.predicate):
                    continue

                # add subject
                worda.append(R.subject.names[0])
                # create subject id
                if not object_dict.has_key(R.subject.names[0]):
                    object_dict[R.subject.names[0]] = len(object_dict)
                subject_ids.append(object_dict[R.subject.names[0]])

                # add object
                wordb.append(R.object.names[0])
                # maintain subject id
                if not object_dict.has_key(R.object.names[0]):
                    object_dict[R.object.names[0]] = len(object_dict)
                object_ids.append(object_dict[R.object.names[0]])

                # add predicate
                predicate.append(R.predicate)
                # maintain predicate id
                if not predicate_dict.has_key(R.predicate):
                    predicate_dict[R.predicate] = len(predicate_dict)
                predicate_ids.append(predicate_dict[R.predicate])

                # maintain number of instances per relationship
                r_str = R.subject.names[0] + "-" + R.predicate + "-" + R.object.names[0]
                if instances.has_key(r_str):
                    instances[r_str] += 1
                else:
                    instances[r_str] = 1


        worda = np.asarray(worda)
        wordb = np.asarray(wordb)
        predicate = np.asarray(predicate)
        predicate_ids = np.asarray(predicate_ids)
        subject_ids = np.asarray(subject_ids)
        object_ids = np.asarray(object_ids)

        # save data
        filtered_data = {"subjects" : worda, "objects" : wordb, "predicates" : predicate, "predicates_ids" : predicate_ids, "subject_ids" : subject_ids, "object_ids" : object_ids, "instances" : instances}
        filtered_data_file = open("filtered_data.p", "wb")
        cPickle.dump(filtered_data, filtered_data_file, 0)
        filtered_data_file.close()


    data = Data(filtered_data)

    return data, nof_predicates

class Data(object):
    """
    Class grouped the list of relationship,, used by language module training.
    """
    def __init__(self, data=None):
        if data!=None:
            self.worda = data["subjects"]
            self.wordb = data["objects"]
            self.predicate = data["predicates"]
            self.predicate_ids = data["predicates_ids"]
            # FIXME - temp until subject_ids will be created
            if data.has_key("subject_ids"):
                self.subject_ids = data["subject_ids"]
            else:
                self.subject_ids = np.asarray([0] * len(self.worda))

            # FIXME - temp until object_ids will be created
            if data.has_key("object_ids"):
                self.object_ids = data["object_ids"]
            else:
                self.object_ids = np.asarray([0] * len(self.wordb))

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
        part1.subject_ids = self.subject_ids[0:part1_max]
        part1.object_ids = self.object_ids[0:part1_max]
        part1.instances = self.instances[0:part1_max]

        part2.worda = self.worda[part1_max + 1:]
        part2.wordb = self.wordb[part1_max + 1:]
        part2.predicate = self.predicate[part1_max + 1:]
        part2.predicate_ids = self.predicate_ids[part1_max + 1:]
        part2.subject_ids = self.subject_ids[part1_max + 1:]
        part2.object_ids = self.object_ids[part1_max + 1:]
        part2.instances = self.instances[part1_max + 1]

        return part1, part2

if __name__ == "__main__":
    prepare_data(1000)