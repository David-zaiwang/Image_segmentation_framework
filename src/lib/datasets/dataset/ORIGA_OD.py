# *coding:utf-8 *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os

import torch.utils.data as data


class ORIGA_OD(data.Dataset):
    num_classes = 2

    def __init__(self, opt, split):
        self.images = []
        self.masks = []
        self.opt = opt

        if split == 'train':
            read_files = os.path.join(opt.data_dir, 'Set_A.txt')
        else:
            read_files = os.path.join(opt.data_dir, 'Set_B.txt')

        self.image_root_folder = os.path.join(opt.data_dir, 'images')
        self.gt_root_folder = os.path.join(opt.data_dir, 'masks')

        self._read_img_mask(self.image_root_folder, self.gt_root_folder, read_files)

    def _read_img_mask(self, image_folder, mask_folder, read_files):
        for img_name in open(read_files):
            image_path = os.path.join(image_folder, img_name.split('.')[0] + '.jpg')
            label_path = os.path.join(mask_folder, img_name.split('.')[0] + '.jpg')

            self.images.append(image_path)
            self.masks.append(label_path)

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)