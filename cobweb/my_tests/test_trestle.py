from random import normalvariate
from pprint import pprint

from cobweb.trestle import TrestleTree

def random_instance(num_objects=5, num_sub_objects=3, num_attributes=2):
    """
    Adapted from https://github.com/cmaclell/concept_formation/blob/006450d47c224bd203220a8a1e98991c4f2d6389/concept_formation/tests/test_structure_mapper.py#L55
    """
    i = {}
    for o in range(num_objects):
        obj = '?rand_obj' + str(o)
        i[obj] = {}

        for a in range(num_attributes):
            attr = 'a' + str(a)
            i[obj][attr] = normalvariate(a, 1)
            # i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

        for so in range(num_sub_objects):
            sobj = '?rand_sobj' + str(so)
            i[obj][sobj] = {}

            for a in range(num_attributes):
                attr = 'a' + str(a)
                i[obj][sobj][attr] = normalvariate(a, 1)
                # i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

    return i

def random_concept(num_instances=1, **kwargs):
    tree = TrestleTree(alpha=1e-6, weight_attr=False, objective=0, children_norm=True, norm_attributes=False)
    pprint(tree.__dict__)
    for i in range(num_instances):
        print("Training concept with instance", i+1)
        inst = random_instance(**kwargs)
        pprint(inst)
        tree.ifit(inst)
    return tree.root

if __name__ == "__main__":
    # test 1
    #i = random_instance(5, 3, 2)
    #pprint(i)
    # test 2
    c = random_concept(5, num_objects=3, num_sub_objects=2, num_attributes=2)
    pprint(c)