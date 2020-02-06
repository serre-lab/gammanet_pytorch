import os
from ops import experiment_tools
from utils import py_utils
import sys
import logging

import torch

torch.backends.cudnn.benchmark = True

def main(cfg):
    logging.info(cfg.pretty())
    exp = experiment_tools.get_experiment(cfg)
    exp.setup()
    exp.run()

if __name__ == "__main__":
    cfg = py_utils.get_config(sys.argv[1:])
    py_utils.setup_logging(cfg.dir)
    
    main(cfg)
