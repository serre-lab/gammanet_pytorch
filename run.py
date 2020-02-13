import os
from ops import experiment_tools
from utils import py_utils
import sys
import logging


def main(cfg):
    logging.info(cfg.pretty())
    exp = experiment_tools.get_experiment(cfg)
    exp.setup()
    exp.run()

if __name__ == "__main__":
    cfg = py_utils.get_config(sys.argv[1:])
    py_utils.setup_logging(cfg.dir)
    print('PID:',os.getpid())
    if 'gpus' in cfg and cfg.gpus is not None:
        py_utils.allocate_gpus(cfg.gpus)
    main(cfg)
