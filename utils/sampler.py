"""
@author: Ankit Dhankhar
@contact: adhankhar@cs.iitr.ac.in
"""
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Sampler


class StratifiedSampler(Sampler):
    """
    Stratified Sampling

    Provide equal representation of target classes in each batch
    """

    def __init__(self, labels):
        self.idx_by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            self.idx_by_label[lb].append(idx)
        self.size = len(labels)

    def __iter__(self):
        idx_list = []
        label_list = []
        for label, indicies in self.idx_by_label.items():
            for idx in indicies:
                idx_list.append(idx)
                label_list.append(label)
        shuffled = Shuffle(idx_list, label_list)
        return iter(shuffled)

    def __len__(self):
        return self.size


def Shuffle(idx_list, lable_list):
    label2idx = defaultdict(list)
    for label, idx in zip(label_list, idx_list):
        label2idx[label].append(idx)
    idxList = []
    idxLoc = []
    for label, idx in label2idx.items():
        idx = fisherYatesShuffle(songs)
        idxList += idx
        idxLoc += get_locs(len(idx))
    return [idxList[index] for index in argsort(idxLoc)]


def argsort(seq):
    return [i for i, j in sorted(enumerate(seq), key=lambda x: x[1])]


def get_locs(n):
    percent = 1.0 / n
    locs = [percent * random.random()]
    last = locs[0]
    for i in range(n - 1):
        value = last + percent * random.uniform(0.8, 1.2)
        locs.append(value)
        last = value
    return locs


def fisherYatesShuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
