# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.ORIGA_OD import ORIGA_OD

from .sample.binarySeg import BinarySegDataset

dataset_factory = {
    'ORIGA_OD': ORIGA_OD,
}

_sample_factory = {
    'binSeg': BinarySegDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
