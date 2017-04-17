import os
import os.path
import numpy as np
import cPickle


def prepare_data(max_nof_predicates):
    """
    Basic implemetation to prepre data for language module training.
    TBD: improve
    :param max_nof_predicates:
    :return:
    """
    file = open("entities.p", "rb")
    entities = cPickle.load(file)
    file.close()

    worda = []
    wordb = []
    predicate = []
    predicate_ids = []
    instances = {}
    predicate_dict = {}
    for entity in entities:
        for R in entity.relationships:
            if len(predicate_dict) >= max_nof_predicates:
                break
            worda.append(R.subject.names[0])
            wordb.append(R.object.names[0])
            predicate.append(R.predicate)
            if not predicate_dict.has_key(R.predicate):
                predicate_dict[R.predicate] = len(predicate_dict)
            predicate_ids.append(predicate_dict[R.predicate])
            r_str = R.subject.names[0] + "-" + R.predicate + "-" + R.object.names[0]
            if instances.has_key(r_str):
                instances[r_str] += 1
            else:
                instances[r_str] = 1


    worda = np.asarray(worda)
    wordb = np.asarray(wordb)
    predicate = np.asarray(predicate)
    predicate_ids = np.asarray(predicate_ids)
    perm1 = np.random.permutation(worda.shape[0])
    perm2 = np.random.permutation(worda.shape[0])

    R1 = Data(worda[perm1], wordb[perm1], predicate[perm1], predicate_ids[perm1], instances)
    R2 = Data(worda[perm2], wordb[perm2], predicate[perm2], predicate_ids[perm1], instances)

    return R1, R2, len(predicate_dict)

class Data(object):
    """
    Class grouped the list of relationship,, used by language module training.
    """
    def __init__(self, worda, wordb, predicate, predicate_ids, instances):
        self.worda = worda
        self.wordb = wordb
        self.predicate = predicate
        self.predicate_ids = predicate_ids
        self.instances = np.zeros(worda.shape[0])
        for i in range(worda.shape[0]):
            self.instances[i] = instances[worda[i] + "-" + predicate[i] + "-" + wordb[i]]

if __name__ == "__main__":
    prepare_data(1000)