"""This module contains simple helper functions """
from __future__ import print_function
import torch
import shutil
import numpy as np
from PIL import Image
import os




def get_accuracy( outputs, labels ):

    ##### expects already concatenated arrays
    # outputs = np.concatenate(outputs, axis = 0)
    # labels  = np.concatenate(labels , axis = 0)

    assert outputs.shape[0] == labels.shape[0]

    preds = np.argmax(outputs, axis=1)
    correct = (preds == labels).sum()
    total = labels.shape[0]
    acc = 1. * correct / total

    return acc


def get_confusion_matrix( outputs, labels , num_classes=7 , normalize = True):

    # outputs = np.concatenate(outputs, axis = 0)
    # labels  = np.concatenate(labels , axis = 0)

    assert outputs.shape[0] == labels.shape[0]
    conf_mat = np.zeros((num_classes, num_classes))

    preds = np.argmax(outputs, axis=1)

    assert preds.shape == labels.shape

    for index in range(labels.shape[0]):
        conf_mat[int(labels[index])][preds[index]] += 1

    if normalize == True:
        conf_mat = np.nan_to_num(
            100. * conf_mat / conf_mat.sum(axis=1, keepdims=True))

    return conf_mat

def get_tblogger(output_dir):
    from logger import Logger
    tb_log_dir = os.path.join(output_dir, 'tb_log')
    # remove previous log_dir if present
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    if not os.path.exists(tb_log_dir):
        mkdirs(tb_log_dir)

    print ('tensorboard --logdir=' +  tb_log_dir + ' --port=6006')
    logger = Logger(tb_log_dir)
    return logger



# http://code.activestate.com/recipes/578933-pasting-python-data-into-a-spread-sheet/
def matrix_to_str(D, transpose=False, replace=None):
    """Construct a string suitable for a spreadsheet.

    D: scalar, 1d or 2d sequence
        For example a list or a list of lists.

    transpose: Bool
        Transpose the data if True.

    replace: tuple or None
        If tuple, it is two strings to pass to the replace
        method. ('toreplace', 'replaceby')

    """

    try:
        D[0]
    except (TypeError, IndexError):
        D = [D]
    try:
        D[0][0]
    except (TypeError, IndexError):
        D = [D]

    if transpose:
        D = zip(*D)

    if not replace:
        # changed here a little bit from the source.
        rows = ['\t'.join(['%.2f' % (v) for v in row]) for row in D]
    else:
        rows = ['\t'.join([str(v).replace(*replace)
                           for v in row]) for row in D]
    S = '\n'.join(rows)
    return S



def save_params( path , config ):
    args = vars(config)
    with open(path, 'wt') as file:
        file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            file.write('%s: %s\n' % (str(k), str(v)))
        file.write('-------------- End ----------------\n')



def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)