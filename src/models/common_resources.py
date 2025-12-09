import logging
import os
import random
import time
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
from torch import optim
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode # TODO: needs install

#### helpers ####

def copy_model_and_optimizer(model, optimizer):
    """
    Copy the model and optimizer states for resetting after adaptation
    """

    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """
    Restore the model and optimizer states from copies
    """

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def merge_from_file(cfg, cfg_file):
    """
    Merges a configuration file into the global TTA configuration
    """
    specific_cfg = CfgNode()
    with g_pathmgr.open(cfg_file, "r") as f:
        specific_cfg.load_cfg(f) # TODO: debug this line
    cfg.merge_from_other_cfg(specific_cfg)

def load_cfg_fom_args(cfg, cfg_file):
    """
    Load config from command line args and set any specified options.
    """

    merge_from_file(cfg_file)

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # logging that config was implemented
    logger = logging.getLogger(__name__)
    logger.info("tta config setup successful")
