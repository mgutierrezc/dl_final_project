from omegaconf import DictConfig, OmegaConf
from src.pipeline.evaluate_unida_tta_models import evaluate_unida_tta_models
from src.pipeline.corruptions_no_cropping import corrupt_datasets
import os, sys, yaml
import traceback
import logging

# in case of glibc++ bug
# NOTE: if error persists, run `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` in terminal
libdir = os.path.join(sys.prefix, "lib")
prev = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{prev}"

# in case of OpenBLAS bug
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#### general setup ####
OmegaConf.register_new_resolver("uuid", lambda: 1)

OUTPUT_DIR = "./output"

def project_pipeline(config: DictConfig):
    """
    Runs the full project pipeline based on the provided configuration.
    """

    import numpy as np
    import json, pickle, random
    from time import sleep
    from tqdm import tqdm
    from src.models.unida_models import load_unida_model
    from src.models.tta_models import setup_tta
    from src.datasets.datasets import get_tta_transforms, read_datalist, preprocess_images_eval, UnidaDataset
    from src.metrics import h_score_and_accuracies_sfunida, h_score_and_accuracies_tasc
    import torch
    from torch.utils.data import DataLoader
    from src.utils import obtain_all_transfers, obtain_config_transfer, obtain_checkpoint_path, obtain_config_dataset, setup_logging
    from src.pipeline.helpers import seed_worker, g_target, initializing_unida_tta_model, calculate_unida_tta_metrics, load_tta_config
    import wandb

    ### setting up determinism

    # setting global seeds
    torch.manual_seed(config.tta_runs.seed)
    torch.cuda.manual_seed(config.tta_runs.seed)
    np.random.seed(config.tta_runs.seed)
    random.seed(config.tta_runs.seed)

    # enforce determinism in torch
    if hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    eval('setattr(torch.backends.cudnn, "benchmark", False)')
    eval('setattr(torch.backends.cudnn, "deterministic", True)')

    ### 1: running evaluation
    if config.current_step == "evaluate_unida_tta_models":
        evaluate_unida_tta_models(config)
    
    ### 2: running corruption evaluation
    if config.current_step == "corrupt_datasets":
        corrupt_datasets(config)

def main(config: DictConfig):
    try:
        project_pipeline(config)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # flush everything
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    # load base config (change path/name as needed)
    base_cfg_path = "./config/config.yaml"  # or wherever your default lives
    base_cfg = OmegaConf.load(base_cfg_path)

    # parse CLI overrides, like current_step=corrupt_datasets
    cli_cfg = OmegaConf.from_cli()

    # merge base + overrides
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # run
    main(cfg)