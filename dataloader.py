import logging
import os

import torch
import torchvision

from torch.utils.data import SubsetRandomSampler, Sampler, Subset
from torchvision.transforms import transforms
from sklearn.model selection import StratifiedShuffleSplit
from theconf import Config as C

cutout = 16
aug = 'default'

from augmentation import *

class CutoutDefault(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img,size(2)
        mask = np.ones((h,w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1:x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies
    
    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

class SubsetSampler(Sampler):
    """
    Sample elements from a given list of indices, without replacement.

    Arguments:
        indices (sequences): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

        