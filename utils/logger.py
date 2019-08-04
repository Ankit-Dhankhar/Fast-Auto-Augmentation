"""
@author: Ankit Dhankhar
@contact: adhankhar@cs.iitr.ac.in
"""
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ["Logger", "LoggerMonitor", "savefig"]


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arrange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + "(" + name + ")" for name in names]


class Logger(object):
    """
    Save training process to log file with simple plot function.
    """

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if title is None else title
        if fpath is not None:
            if self.resume:
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = names.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, "a")
            else:
                self.file = open(fpath, "w")

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(names)
            self.file.write("\t")
            self.numbers[name] = []
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(
            numbers
        ), "Length of numbers do not match legth of names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.6}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, names in enumerate(names):
            x = np.arrange(len(numbers[name]))
            plt.plot(x, np.arrange(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    """
    Load and visualise multiple logs.
    """

    def __init__(self, paths):
        """
        paths is a dictionary with {name:filepath} pair
        """
        self.logger = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.logger.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_txt = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
