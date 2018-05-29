# coding=utf-8
import math
import os
import random
import re
from enum import Enum
from typing import Dict, Union, Tuple, Set
from warnings import warn

import torch

from e2e import E2E, E2ESet


# NOTE: 'familyFriendly' and 'area' are not included because they have only 2 possible values, that is too few to split
class E2EAttribute(Enum):
    CUSTOMER_RATING = 'customer rating'
    EAT_TYPE = 'eatType'
    FOOD = 'food'
    NAME = 'name'
    NEAR = 'near'
    PRICE_RANGE = 'priceRange'
    # AREA = 'area'
    # FAMILY_FRIENDLY = 'familyFriendly'


class E2EAttrSplit(E2E):
    _partitions_file = 'partitions.pt'

    def __init__(self, root, which_set: E2ESet, vocabulary_class, attribute: E2EAttribute,
                 training_ratio=0.8, tolerance=0.01):
        if not 0 < training_ratio < 1:
            raise ValueError('training_ratio must be between 0 and 1!')
        _check_which_set(which_set)

        super(E2EAttrSplit, self).__init__(root, E2ESet.ALL_IN_ONE, vocabulary_class)

        # Build MR dictionaries
        mr_dict = [_mr_to_dict(self.to_string(example)) for example in self.mr]

        # Only use MRs that contain the selected attribute
        attribute = attribute.value
        data_zip = zip(self.mr, mr_dict, self.ref)
        data_zip = list(filter(lambda z: attribute in z[1].keys(), data_zip))
        self.mr, mr_dict, self.ref = [list(z) for z in zip(*data_zip)]

        # Partition values according to the ratio
        self.train_values, self.dev_values = self._partition_values(mr_dict, attribute, training_ratio, tolerance)

        # Build the two MR/REF partitions
        train_zip = list(filter(lambda z: z[1][attribute] in self.train_values, data_zip))
        train_zip.sort(key=lambda z: z[0])
        dev_zip = list(filter(lambda z: z[1][attribute] in self.dev_values, data_zip))
        dev_zip.sort(key=lambda z: z[0])
        self.mr_train, _, self.ref_train = [list(z) for z in zip(*train_zip)]
        self.mr_dev, _, self.ref_dev = [list(z) for z in zip(*dev_zip)]

        # Choose train/dev set according to which_set
        self._set_chooser = {
            E2ESet.TRAIN: (self.mr_train, self.ref_train),
            E2ESet.DEV:   (self.mr_dev, self.ref_dev)
        }
        self.choose_set(which_set)

    # NOTE: this is the Subset Sum Problem - and it's NP-Hard! D:
    # Approximate algorithms exist: maybe-TODO use one of them instead of this fanciful method?
    def _partition_values(self, mr_dict: Dict[str, float], attribute: E2EAttribute,
                          partition_ratio: float, tolerance: float) -> Tuple[Set[str], Set[str]]:
        # This file contains all the successful partitions
        partitions_file_name = os.path.join(self.root, self.processed_folder, self._partitions_file)
        if os.path.isfile(partitions_file_name):
            partitions = torch.load(partitions_file_name)
            # If this partition has already been done, just use it
            if (attribute, partition_ratio, tolerance) in partitions.keys():
                return partitions[(attribute, partition_ratio, tolerance)]
        else:
            partitions = dict()

        values_distribution = _count_values(mr_dict, attribute)

        train_values = set()
        dev_values = set(values_distribution.keys())
        current_ratio = 0
        tolerance_percent = tolerance * partition_ratio

        # Using a counter to avoid infinite loops:
        # the tolerance is doubled every max_cnt iterations
        cnt = 0
        max_cnt = 2 * len(values_distribution)
        more_tolerance = False
        while abs(current_ratio - partition_ratio) > tolerance_percent or \
                len(dev_values) == 0 or len(train_values) == 0:
            if current_ratio < partition_ratio:
                # randomly transfer an element from dev_values to train_values
                element = random.sample(dev_values, 1)[0]
                dev_values.remove(element)
                train_values.add(element)
                # Update current ratio
                current_ratio += values_distribution[element]
            else:
                # randomly transfer an element from train_values to dev_values
                element = random.sample(train_values, 1)[0]
                train_values.remove(element)
                dev_values.add(element)
                # Update current ratio
                current_ratio -= values_distribution[element]

            cnt += 1
            if cnt == max_cnt:
                more_tolerance = True
                cnt = 0
                tolerance_percent *= 2
        if more_tolerance:
            warn('The partitioning was unexpectedly complicated! Requested ratio was ' +
                 '({:.1f} ± {:.1f})%,'.format(100 * partition_ratio, 100 * tolerance * partition_ratio) +
                 'but {:.2f}% is all I can do :('.format(100 * current_ratio))
        else:
            # If this is a good partition, save it
            partitions[(attribute, partition_ratio, tolerance)] = (train_values, dev_values)
            torch.save(partitions, partitions_file_name)

        return train_values, dev_values

    def choose_set(self, which_set: E2ESet):
        _check_which_set(which_set)
        self.which_set = which_set
        self.mr, self.ref = self._set_chooser[which_set]


def _check_which_set(e2e_set):
    if e2e_set not in [E2ESet.TRAIN, E2ESet.DEV]:
        raise ValueError('Only SetType.TRAIN and SetType.DEV are allowed on this class!')


def _count_values(dict_list, key):
    step = 1 / len(dict_list)
    counter = dict()
    for mr in dict_list:
        value = mr[key]
        if value in counter.keys():
            counter[value] += step
        else:
            counter[value] = step
    assert math.isclose(sum(counter.values()), 1)
    return counter


def _mr_to_dict(mr: Union[str, torch.Tensor, list]) -> Dict[str, str]:
    attributes = {}
    for attribute in mr.split(', '):
        key_val = re.split(' *[\[\]] *', attribute)
        attributes[key_val[0]] = key_val[1]
    return attributes
