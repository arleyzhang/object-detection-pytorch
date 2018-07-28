from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from ast import literal_eval
import numpy as np
import os.path as osp
import six
import yaml


# Note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg = __C


# ---------------------------------------------------------------------------- #
# General options
# ---------------------------------------------------------------------------- #
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.CFG_ROOT = osp.abspath(osp.join(__C.ROOT_DIR, 'cfgs'))
__C.CUDA_VISIBLE_DEVICES = '0,1,2,3'


# ---------------------------------------------------------------------------- #
# Log options
# ---------------------------------------------------------------------------- #
__C.LOG = AttrDict()
__C.LOG.ROOT_DIR = './experiments/models/ssd_voc'


# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()
# -- dataset specific option
# name of the dataset
__C.DATASET.NAME = 'VOC0712'
__C.DATASET.SUB_DIR = 'VOCdevkit'
# path of the dataset
__C.DATASET.DATASET_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data', __C.DATASET.SUB_DIR))
# train set scope
__C.DATASET.TRAIN_SETS = [('2007', 'trainval'), ('2012', 'trainval')]
# test set scope
# __C.DATASET.TEST_SETS = [('2007', 'test')]  # [('2007', 'test')]
__C.DATASET.TEST_SETS = [('2007', 'test')]  # [('2007', 'test')]
# class number in the dataset
__C.DATASET.NUM_CLASSES = 20

# -- augmentation option
# image size (h, w)
__C.DATASET.IMAGE_SIZE = (300, 300)
# image expand probability during train
__C.DATASET.PROB = 0.6
# image mean
__C.DATASET.PIXEL_MEANS = (104, 117, 123)

# -- dataloader option
# train batch size
__C.DATASET.TRAIN_BATCH_SIZE = 32
# test batch size
__C.DATASET.EVAL_BATCH_SIZE = 32
# number of workers to extract datas
__C.DATASET.NUM_WORKERS = 8
# number of eval images, 0 means total test set
__C.DATASET.NUM_EVAL_PICS = 0


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
# Name of the base net used to extract the features
__C.MODEL.BASE = 'vgg16'
# Name of the model used to detect boundingbox
__C.MODEL.SSD_TYPE = 'SSD'
# Prior box type
__C.MODEL.PRIOR_TYPE = 'PriorBoxSSD'
# Prior box type
__C.MODEL.NUM_PRIOR = None
# number of the class for the model
__C.MODEL.NUM_CLASSES = __C.DATASET.NUM_CLASSES + 1
# number of the class for the model
__C.MODEL.IMAGE_SIZE = __C.DATASET.IMAGE_SIZE
# STEPS for the proposed bounding box, if empty the STEPS = image_size / feature_map_size

# -- Prior box option
__C.MODEL.STEPS = [8, 16, 32, 64, 100, 300]
# a list from min value to max value
__C.MODEL.MIN_SIZES = [30, 60, 111, 162, 213, 264]
__C.MODEL.MAX_SIZES = [60, 111, 162, 213, 264, 315]
# ASPECT_RATIOS for the proposed bounding box, 1 is default contains
__C.MODEL.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
__C.MODEL.VARIANCE = [0.1, 0.2]
__C.MODEL.CLIP = True
__C.MODEL.FLIP = True


# ---------------------------------------------------------------------------- #
# Evaluation options
# ---------------------------------------------------------------------------- #
__C.EVAL = AttrDict()
# __C.EVAL.BATCH_SIZE = __C.TRAIN.BATCH_SIZE
# __C.EVAL.TEST_SCOPE = [0, 300]


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()
# # Minibatch size
__C.TRAIN.MAX_ITER = 120000


# ---------------------------------------------------------------------------- #
# optimizer options
# ---------------------------------------------------------------------------- #
__C.TRAIN.OPTIMIZER = AttrDict()
# type of the optimizer
__C.TRAIN.OPTIMIZER.OPTIMIZER = 'sgd'
# Initial learning rate
__C.TRAIN.OPTIMIZER.LR = 1e-3
# Momentum
__C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Weight decay, for regularization
__C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4


# ---------------------------------------------------------------------------- #
# lr_scheduler options
# ---------------------------------------------------------------------------- #
__C.TRAIN.LR_SCHEDULER = AttrDict()
# type of the LR_SCHEDULER
__C.TRAIN.LR_SCHEDULER.SCHEDULER = 'step'
# Step size for reducing the learning rate
__C.TRAIN.LR_SCHEDULER.STEPS = (80000, 100000, 120000)
# Factor for reducing the learning rate
__C.TRAIN.LR_SCHEDULER.GAMMA = 0.1


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)
    update_cfg()


def update_cfg():
    # TODO this is error prone
    __C.DATASET.DATASET_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data', __C.DATASET.SUB_DIR))
    __C.MODEL.NUM_CLASSES = __C.DATASET.NUM_CLASSES + 1


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a