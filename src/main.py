# *coding:utf-8 *
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from _init_paths import __init_path

import torch
import torch.utils

from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model, save_model
from trains.train_factory import train_factory

from opts import opts


def main(opt):
    # Completely reproducible results are not guaranteed across PyTorch releases, \
    # individual commits, or different platforms. Furthermore, results may not be reproducible \
    # between CPU and GPU executions, even when using identical seeds.
    # We can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(opt.seed)

    # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，\
    # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。\
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，\
    # 其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = get_dataset(opt.dataset, opt.task)

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')

    model = create_model(opt.model_name)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]

    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

if __name__ == '__main__':
    __init_path()
    print(sys.path)
