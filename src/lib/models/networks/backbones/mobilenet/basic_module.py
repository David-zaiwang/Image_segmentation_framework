from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original pytorch repo
    It ensures that all layers have a channel number that is divisable by 8
    it can be seen here:
    https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3,
                 stride=1, group=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size,
                      stride, padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(in_planes)
        )
        

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
