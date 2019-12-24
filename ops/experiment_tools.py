import os
from utils import py_utils

def get_experiment(cfg):
    exp_module = py_utils.import_module(cfg.import_prepath)
    exp = getattr(exp_module, cfg.import_class)(cfg)
    return exp
