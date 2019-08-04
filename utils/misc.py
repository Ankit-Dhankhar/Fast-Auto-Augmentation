"""
@author: Ankit Dhankhar
@contact: adhankhar@cs.iitr.ac.in
"""
import errno
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = [
    "get_mean_and_std",
    "init_params",
    "mkdir_p",
    "AverageMeter",
    "save_checkpoint",
]


def get_mean_and_std(dataset):
    """
    Compute the mean and std values of dataset.
    """
    dataloader = trainloader = torch.utils.DataLoader(
        dataset, batch_size=1, shuffle=True, num_worker=2
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computinf mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += input[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """
    Initialize layer parameters.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    """
    Make a direction if it doesn't exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """
    Computes and stores the average and current values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoiny="checkpoint", filename="checkpoint.pth"):
    filename = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth"))
