"""
This is a quick adaption of https://github.com/cmaclell/cobweb
"""

from pprint import pprint

from cobweb.cobweb import CobwebTree
from cobweb.structure_mapper import StructureMapper
from cobweb.preprocessor import SubComponentProcessor, Flattener, Pipeline, NameStandardizer

class TrestleTree(CobwebTree):
    def __init__(self, **kwargs):
        """
        The tree constructor.
        """
        super(TrestleTree, self).__init__(**kwargs)
        self.gensym_counter = 0

    def clear(self):
        """
        Clear the tree but keep initialization parameters
        """
        self.gensym_counter = 0
        super(TrestleTree, self).clear()

    def gensym(self):
        """
        Generates unique names for naming renaming apart objects.

        :return: a unique object name
        :rtype: '?o'+counter
        """
        self.gensym_counter += 1
        return '?o' + str(self.gensym_counter)

    def _sanity_check_instance(self, instance):
        """
        Checks the attributes of an instance to ensure they are properly
        subscriptable types and throws an excpetion if they are not.
        Lots of sub-processes in the structure mapper freak out if you have
        non-str non-tuple attributes so I decided it was best to do a one
        time check at the first call to transform.
        """
        for attr in instance:
            try:
                hash(attr)
                attr[0]
            except Exception:
                raise ValueError('Invalid attribute: '+str(attr) +
                                 ' of type: ' + str(type(attr)) +
                                 ' in instance: ' + str(instance) +
                                 ',\n' + type(self).__name__ +
                                 ' only works with hashable and' +
                                 ' subscriptable attributes (e.g., strings).')
            if isinstance(attr, tuple):
                self._sanity_check_relation(attr, instance)
            if isinstance(instance[attr], dict):
                self._sanity_check_instance(instance[attr])
            else:
                try:
                    hash(instance[attr])
                except Exception:
                    raise ValueError('Invalid value: ' + str(instance[attr]) +
                                     ' of type: ' + str(type(instance[attr])) +
                                     ' in instance: ' + str(instance) +
                                     ',\n' + type(self).__name__ +
                                     ' only works with hashable values.')

    def _sanity_check_relation(self, relation, instance):
        for v in relation:
            try:
                v[0]
            except Exception:
                raise(ValueError('Invalid relation value: ' + str(v) +
                                 ' of type: ' + str(type(v)) +
                                 ' in instance: ' + str(instance) +
                                 ',\n' + type(self).__name__ +
                                 'requires that values inside relation' +
                                 ' tuples be of type str or tuple.'))
            if isinstance(v, tuple):
                self._sanity_check_relation(v, instance)

    def _trestle_categorize(self, instance):
        """
        The structure maps the instance, categorizes the matched instance, and
        returns the resulting concept.

        :param instance: an instance to be categorized into the tree.
        :type instance: {a1:v1, a2:v2, ...}
        :return: A concept describing the instance
        :rtype: concept
        """
        preprocessing = Pipeline(NameStandardizer(self.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(self.root))
        temp_instance = preprocessing.transform(instance)
        self._sanity_check_instance(temp_instance)
        return self._cobweb_categorize(temp_instance)

    def infer_missing(self, instance, choice_fn="most likely",
                      allow_none=True):
        """
        Given a tree and an instance, returns a new instance with attribute
        values picked using the specified choice function (either "most likely"
        or "sampled").

        .. todo:: write some kind of test for this.

        :param instance: an instance to be completed.
        :type instance: :ref:`Instance<instance-rep>`
        :param choice_fn: a string specifying the choice function to use,
            either "most likely" or "sampled".
        :type choice_fn: a string
        :param allow_none: whether attributes not in the instance can be
            inferred to be missing. If False, then all attributes will be
            inferred with some value.
        :type allow_none: Boolean
        :return: A completed instance
        :rtype: instance
        """
        preprocessing = Pipeline(NameStandardizer(self.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(self.root))

        temp_instance = preprocessing.transform(instance)
        concept = self._cobweb_categorize(temp_instance)

        for attr in concept.attrs('all'):
            if attr in temp_instance:
                continue
            val = concept.predict(attr, choice_fn, allow_none)
            if val is not None:
                temp_instance[attr] = val

        temp_instance = preprocessing.undo_transform(temp_instance)
        return temp_instance

    def categorize(self, instance):
        """
        Sort an instance in the categorization tree and return its resulting
        concept.

        The instance is passed down the the categorization tree according to
        the normal cobweb algorithm except using only the new and best
        opperators and without modifying nodes' probability tables. **This does
        not modify the tree's knowledge base** for a modifying version see
        :meth:`TrestleTree.ifit`

        This version differs from the normal :meth:`CobwebTree.categorize
        <cobweb.cobweb.CobwebTree.categorize>` by structure
        mapping instances before categorizing them.

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode

        .. seealso:: :meth:`TrestleTree.trestle`
        """
        return self._trestle_categorize(instance)

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.
        
        The instance is passed down the tree and updates each node to
        incorporate the instance. **This modifies the tree's knowledge** for a
        non-modifying version see: :meth:`TrestleTree.categorize`.

        This function is similar to :meth:`Cobweb.ifit
        <cobweb.cobweb.CobwebTree.ifit>` The key difference
        between trestle and cobweb is that trestle performs structure mapping
        (see: :meth:`structure_map
        <cobweb.structure_mapper.StructureMapper.transform>`) before
        proceeding through the normal cobweb algorithm.

        :param instance: an instance to be categorized into the tree.
        :type instance: :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebNode
        """
        preprocessing = Pipeline(
            NameStandardizer(self.gensym),
            Flattener(), 
            SubComponentProcessor(),
            StructureMapper(self.root),
        )
        temp_instance = preprocessing.transform(instance)
        self._sanity_check_instance(temp_instance)
        pprint(temp_instance)
        #pprint(type(super()))
        #pprint(super())
        return super().ifit(temp_instance)