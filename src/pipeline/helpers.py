import numpy as np
import os, torch
import pickle as pkl
import random
import logging
import json
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import wordnet
from src.models.tta_models import setup_tta
from src.models.unida_models import load_unida_model
from src.utils import obtain_checkpoint_path
from src.datasets.datasets import get_tta_transforms
from src.metrics import h_score_and_accuracies_sfunida, h_score_and_accuracies_tasc

def seed_worker(worker_id, run_seed=2020):
    """
    Make numpy/python in each worker deterministic (independent of global RNG)
    """

    worker_seed = (run_seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def initializing_unida_tta_model(model_key, num_classes, dataset_config, dataset_key, source, target, setting_key, model_path, tta_method, tta_config, 
                                 consistency_filtering=None, type_cf_booster=None, memory_reset=None, risk_memory_size=None, source_loader=None, 
                                 tta_enabled=None):
    """
    Finds checkpoint and initializes UniDA model
    """

    logging.info("initializing unida model")

    if model_key == "tasc":
        checkpoint_path, target_classnames_gsearch, cluster_params_path = obtain_checkpoint_path(model_key, dataset_key, source, target, setting_key, model_path)
    else:
        checkpoint_path = obtain_checkpoint_path(model_key, dataset_key, source, target, setting_key, model_path)
        
    # model init
    logging.info(f"initializing model: {model_key}")
    logging.info(f"source_classnames: {dataset_config['source_classnames']}")
    if model_key == "tasc":
        unida_model = load_unida_model(model_key, checkpoint_path, num_classes=num_classes, backbone_source="pth", 
                                        cluster_params_path=cluster_params_path,
                                        target_classnames_gsearch=target_classnames_gsearch,
                                        source_classnames=dataset_config["source_classnames"])
    else:
        unida_model = load_unida_model(model_key, checkpoint_path, num_classes=num_classes, backbone_source="pth")
    logging.info(f"model initialized in device: {unida_model.device}")

    unida_model.eval()  # set model to evaluation mode

    # setting tta if enabled
    if tta_enabled:
        logging.info(f"initializing_unida_tta_model with tta_config: {tta_config}")
        logging.info(f"source_loader: {source_loader}")
        if model_key == "tasc":
            unida_model = setup_tta(base_model=unida_model, tta_method=tta_method, cfg=tta_config, 
                                    model_key=model_key, checkpoint_path=checkpoint_path, 
                                    target_classnames_gsearch=target_classnames_gsearch, 
                                    cluster_params_path=cluster_params_path, 
                                    source_classnames=dataset_config["source_classnames"],
                                    consistency_filtering=consistency_filtering,
                                    type_cf_booster=type_cf_booster,
                                    memory_reset=memory_reset,
                                    risk_memory_size=risk_memory_size,
                                    source_loader=source_loader)
        else:
            unida_model = setup_tta(unida_model, tta_method, tta_config, model_key, checkpoint_path,
                                    consistency_filtering=consistency_filtering,
                                    type_cf_booster=type_cf_booster,
                                    memory_reset=memory_reset,
                                    risk_memory_size=risk_memory_size,
                                    source_loader=source_loader)
        logging.info(f"setting tta method {tta_method} on model adapted using {model_key}")        

    logging.info("unida model initialized")
    return unida_model

def load_tta_config(tta_method, grid_search_enabled=False, config=None, model_path=None, num_classes=None, source_class_list=None, target_class_list=None, model_key=None,
                    dataset_key=None, source=None, target=None, setting_key=None, transfer=None):
    """
    Loads TTA configuration from a file based on the tta_method
    """

    tta_params = None # placeholder for string w/ tta hyperparameters
    
    tta_config = {} # placeholder for tta config dict
    tta_config["optim"] = {}
    tta_config[tta_method] = {}

    # current eval configuration
    tta_config["current_eval"] = {}
    tta_config["current_eval"]["model_key"] = model_key
    tta_config["current_eval"]["model_path"] = model_path
    tta_config["current_eval"]["dataset_key"] = dataset_key
    tta_config["current_eval"]["source"] = source
    tta_config["current_eval"]["target"] = target
    tta_config["current_eval"]["setting_key"] = setting_key
    tta_config["current_eval"]["num_classes"] = num_classes
    tta_config["current_eval"]["batch_size"] = config.test_batch_size
    tta_config["current_eval"]["corruptions_path"] = config.data_paths.corruptions
    tta_config["current_eval"]["labels_file_name"] = config.data_paths.labels_file_name
    tta_config["current_eval"]["use_unk_classifier"] = config.tta_runs.use_unk_classifier
    tta_config["current_eval"]["source_class_list"] = source_class_list
    tta_config["current_eval"]["target_class_list"] = target_class_list

    if grid_search_enabled: # for grid search: using hyperparameters from experiment config
        tta_config["optim"]["lr"] = config.tta_runs.learning_rate
        
        if tta_method == "stamp":
            tta_config["optim"]["rho"] = config.tta_runs.rho
            tta_config[tta_method]["alpha"] = config.tta_runs.alpha
        elif tta_method == "owttt":
            tta_config[tta_method]["ce_scale"] = config.tta_runs.ce_scale
            tta_config[tta_method]["da_scale"] = config.tta_runs.da_scale
            tta_config[tta_method]["delta"] = config.tta_runs.delta
            tta_config[tta_method]["queue_length"] = config.tta_runs.queue_length
            tta_config[tta_method]["max_prototypes"] = config.tta_runs.max_prototypes
        elif tta_method == "tent":
            tta_config["optim"]["beta"] = config.tta_runs.beta
            tta_config["optim"]["wd"] = config.tta_runs.wd
        elif tta_method == "eata":
            tta_config["optim"]["beta"] = config.tta_runs.beta
            tta_config["optim"]["wd"] = config.tta_runs.wd
            tta_config[tta_method]["fisher_alpha"] = config.tta_runs.fisher_alpha
            tta_config[tta_method]["d_margin"] = config.tta_runs.d_margin
        elif tta_method == "sotta":
            tta_config["optim"]["beta"] = config.tta_runs.beta
            tta_config["optim"]["wd"] = config.tta_runs.wd
            tta_config["optim"]["rho"] = config.tta_runs.rho
            tta_config[tta_method]["threshold"] = config.tta_runs.threshold
        elif tta_method == "sotta_v2":
            tta_config["optim"]["beta"] = config.tta_runs.beta
            tta_config["optim"]["wd"] = config.tta_runs.wd
            tta_config["optim"]["rho"] = config.tta_runs.rho
            tta_config[tta_method]["threshold"] = config.tta_runs.threshold
            tta_config[tta_method]["neg_mem_weight"] = config.tta_runs.neg_mem_weight
        elif tta_method == "gmm":
            tta_config[tta_method]["red_feature_dim"] = config.tta_runs.red_feature_dim
            tta_config[tta_method]["p_reject"] = config.tta_runs.p_reject
            tta_config[tta_method]["N_init"] = config.tta_runs.N_init
            tta_config[tta_method]["augmentation"] = config.tta_runs.augmentation
            tta_config[tta_method]["lam"] = config.tta_runs.lam
            tta_config[tta_method]["temperature"] = config.tta_runs.temperature
        elif tta_method == "unida_tta":
            tta_config["optim"]["rho"] = config.tta_runs.rho
            tta_config[tta_method]["kl_weight"] = config.tta_runs.kl_weight
        elif "oracle" in tta_method:
            tta_config[tta_method]["params_oracle"] = config.tta_runs.params_oracle
            tta_config[tta_method]["neg_mem_weight"] = config.tta_runs.neg_mem_weight
            tta_config[tta_method]["kl_weight"] = config.tta_runs.kl_weight

        logging.info(f"tta_config (grid search): {tta_config}")
    
    else: # after grid search: use best set of hyperparameters
        tta_hyperparams_path = config.tta_runs.config_paths.grid_search_hyperparams
        with open(tta_hyperparams_path, "r") as tta_hyperparams_file:
            tta_hyperparams = json.load(tta_hyperparams_file)

        # optimizer parameters
        tta_config["optim"]["lr"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["lr"]
        if tta_method == "stamp":
            tta_config["optim"]["rho"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["rho"]
            tta_config[tta_method]["alpha"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["alpha"]
            tta_params = f"lr_{tta_config['optim']['lr']}_rho_{tta_config['optim']['rho']}_alpha_{tta_config[tta_method]['alpha']}_bs_{config.test_batch_size}"
        elif tta_method == "owttt":
            tta_config[tta_method]["ce_scale"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["ce_scale"]
            tta_config[tta_method]["da_scale"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["da_scale"]
            tta_config[tta_method]["delta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["delta"]
            tta_config[tta_method]["queue_length"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["queue_length"]
            tta_config[tta_method]["max_prototypes"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["max_prototypes"]
            tta_params = f"lr_{tta_config['optim']['lr']}_bs_{config.test_batch_size}_ce_{tta_config[tta_method]['ce_scale']}_da_{tta_config[tta_method]['da_scale']}_delta_{tta_config[tta_method]['delta']}_ql_{tta_config[tta_method]['queue_length']}_mp_{tta_config[tta_method]['max_prototypes']}"
        elif tta_method == "tent":
            tta_config["optim"]["beta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["beta"]
            tta_config["optim"]["wd"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["wd"]
        elif tta_method == "eata":
            tta_config["optim"]["beta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["beta"]
            tta_config["optim"]["wd"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["wd"]
            tta_config[tta_method]["fisher_alpha"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["fa"]
            tta_config[tta_method]["d_margin"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["dm"]
            tta_params = f"lr_{tta_config['optim']['lr']}_beta_{tta_config['optim']['beta']}_wd_{tta_config['optim']['wd']}_fa_{tta_config[tta_method]['fisher_alpha']}_dm_{tta_config[tta_method]['d_margin']}_bs_{config.test_batch_size}"
        elif tta_method == "sotta":
            tta_config["optim"]["beta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["beta"]
            tta_config["optim"]["wd"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["wd"]
            tta_config["optim"]["rho"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["rho"]
            tta_config[tta_method]["threshold"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["th"]
            tta_params = f"lr_{tta_config['optim']['lr']}_beta_{tta_config['optim']['beta']}_wd_{tta_config['optim']['wd']}_rho_{tta_config['optim']['rho']}_th_{tta_config[tta_method]['threshold']}_bs_{config.test_batch_size}"
        elif tta_method == "sotta_v2":
            tta_config["optim"]["beta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["beta"]
            tta_config["optim"]["wd"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["wd"]
            tta_config["optim"]["rho"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["rho"]
            tta_config[tta_method]["threshold"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["th"]
            tta_config[tta_method]["neg_mem_weight"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["neg_mem_weight"]
            tta_params = f"lr_{tta_config['optim']['lr']}_beta_{tta_config['optim']['beta']}_wd_{tta_config['optim']['wd']}_rho_{tta_config['optim']['rho']}_th_{tta_config[tta_method]['threshold']}_bs_{config.test_batch_size}"
        elif tta_method == "gmm":
            tta_config[tta_method]["red_feature_dim"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["red_feature_dim"]
            tta_config[tta_method]["p_reject"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["p_reject"]
            tta_config[tta_method]["N_init"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["N_init"]
            tta_config[tta_method]["augmentation"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["augmentation"]
            tta_config[tta_method]["lam"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["lam"]
            tta_config[tta_method]["temperature"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["temperature"]
            tta_params = f"lr_{tta_config['optim']['lr']}_bs_{config.test_batch_size}_redfd_{tta_config[tta_method]['red_feature_dim']}_prej_{tta_config[tta_method]['p_reject']}_Ninit_{tta_config[tta_method]['N_init']}_aug_{tta_config[tta_method]['augmentation']}_lam_{tta_config[tta_method]['lam']}_temp_{tta_config[tta_method]['temperature']}"
        elif tta_method == "unida_tta":
            tta_config["optim"]["rho"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["rho"]
            tta_config["unida_tta"]["kl_weight"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["kl"]
            tta_params = f"lr_{tta_config['optim']['lr']}_rho_{tta_config['optim']['rho']}_bs_{config.test_batch_size}_kl_{tta_config['unida_tta']['kl_weight']}"
        elif "oracle" in tta_method:
            tta_config["optim"]["beta"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["beta"]
            tta_config["optim"]["wd"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["wd"]
            tta_config[tta_method]["neg_mem_weight"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["neg_mem_weight"]
            tta_config[tta_method]["kl_weight"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["kl_weight"]
            tta_config[tta_method]["params_oracle"] = tta_hyperparams[model_key][dataset_key][setting_key][transfer][tta_method]["params_oracle"]
            tta_params = f"lr_{tta_config['optim']['lr']}_negw_{tta_config[tta_method]['neg_mem_weight']}_kl_{tta_config[tta_method]['kl_weight']}_params_{tta_config[tta_method]['params_oracle']}_bs_{config.test_batch_size}"

        logging.info(f"tta_config (no grid search): {tta_config}")

    return tta_config, tta_params

def wandb_assign_project_name(config, corruption, corruption_type, severity, tta_enabled, tta_method, 
                              memory_reset, risk_memory_size, tta_grid_search, dataset_key, model_key,
                              use_unk_classifier):
    """
    Assigns a project name to the wandb run based on the model, dataset, transfer, setting, 
    corruption and tta method
    """

    tta_grid_search_params = None

    if corruption and tta_enabled:
        project_name = config.logging.wandb_project_name + "_corruption_" + corruption_type + f"_sev_{severity}" + f"_tta_{tta_method}"

        project_name += "norm_and_lora"

        if memory_reset:
            project_name += "_mem_reset"

        if risk_memory_size:
            project_name += f"_rms_{risk_memory_size}"

        if use_unk_classifier:
            project_name += f"_unk_cls_updt"
        else:
            project_name += f"_no_unk_cls"
        
        if tta_grid_search:
            # general parameters
            lr_str = str(config.tta_runs.learning_rate).replace(".", "")
            
            if tta_method == "stamp":
                rho_str = str(config.tta_runs.rho).replace(".", "")
                alpha_str = str(config.tta_runs.alpha).replace(".", "")            
                tta_grid_search_params = f"lr_{lr_str}_rho_{rho_str}_alpha_{alpha_str}_bs_{config.test_batch_size}"
            elif tta_method == "owttt":
                da_str = str(config.tta_runs.da_scale).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_bs_{config.test_batch_size}_da_{da_str}"
            elif tta_method == "tent":
                beta_str = str(config.tta_runs.beta).replace(".", "")
                wd_str = str(config.tta_runs.wd).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_beta_{beta_str}_wd_{wd_str}_bs_{config.test_batch_size}"
            elif tta_method == "eata":
                beta_str = str(config.tta_runs.beta).replace(".", "")
                wd_str = str(config.tta_runs.wd).replace(".", "")
                fisher_alpha_str = str(config.tta_runs.fisher_alpha).replace(".", "")
                d_margin_str = str(config.tta_runs.d_margin).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_beta_{beta_str}_wd_{wd_str}_fa_{fisher_alpha_str}_dm_{d_margin_str}_bs_{config.test_batch_size}"
            elif tta_method == "sotta":
                beta_str = str(config.tta_runs.beta).replace(".", "")
                wd_str = str(config.tta_runs.wd).replace(".", "")
                threshold_str = str(config.tta_runs.threshold).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_beta_{beta_str}_wd_{wd_str}_th_{threshold_str}_bs_{config.test_batch_size}"
            elif tta_method == "sotta_v2":
                beta_str = str(config.tta_runs.beta).replace(".", "")
                wd_str = str(config.tta_runs.wd).replace(".", "")
                threshold_str = str(config.tta_runs.threshold).replace(".", "")
                neg_mem_weight_str = str(config.tta_runs.neg_mem_weight).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_beta_{beta_str}_wd_{wd_str}_th_{threshold_str}_nmw_{neg_mem_weight_str}_bs_{config.test_batch_size}"
            elif tta_method == "gmm":
                red_feature_dim_str = str(config.tta_runs.red_feature_dim).replace(".", "")
                p_reject_str = str(config.tta_runs.p_reject).replace(".", "")
                N_init_str = str(config.tta_runs.N_init).replace(".", "")
                augmentation_str = str(int(config.tta_runs.augmentation)) # bool to int
                lam_str = str(config.tta_runs.lam).replace(".", "")
                temperature_str = str(config.tta_runs.temperature).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_bs_{config.test_batch_size}_redfd_{red_feature_dim_str}_prej_{p_reject_str}_Ninit_{N_init_str}_aug_{augmentation_str}_lam_{lam_str}_temp_{temperature_str}"
            elif tta_method == "unida_tta":
                rho_str = str(config.tta_runs.rho).replace(".", "")
                kl_str = str(config.tta_runs.kl_weight).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_rho_{rho_str}_kl_{kl_str}_bs_{config.test_batch_size}"
            elif "oracle" in tta_method:
                neg_w_str = str(config.tta_runs.neg_mem_weight).replace(".", "")
                kl_str = str(config.tta_runs.kl_weight).replace(".", "")
                tta_grid_search_params = f"lr_{lr_str}_negw_{neg_w_str}_kl_{kl_str}_bs_{config.test_batch_size}"

            project_name += f"_dset_{dataset_key}_model_{model_key}_grid_search"
        else:
            pass

    elif corruption and not tta_enabled:
        project_name = config.logging.wandb_project_name + "_corruption_" + corruption_type + f"_sev_{severity}" + "_tta_disabled" 
    else:
        project_name = config.logging.wandb_project_name + "_baseline"
    
    if config.debug:
        project_name = "debug_" + project_name
    
    return project_name, tta_grid_search_params

def calculate_unida_tta_metrics(images, model_key, tta_enabled, tta_method, unida_model, pred_probs_stack, unknown_scores_dict_stack,
                                gt_label_stack, target_class_list, gt_labels, img_indices, unida_dataset, 
                                dataset_key, open_flag, tta_num_augmentations):
    """
    Obtains the H-score, known accuracy, unknown accuracy and per class accuracy
    for a given model and dataset
    """

    logger = logging.getLogger(__name__)
    logger.info(f"unida_model.device = {unida_model.device}")
    images = images.to(unida_model.device)
    if not tta_enabled:
        with torch.no_grad():
            if model_key == "tasc":
                predicted_probs, unknown_scores_dict = unida_model(images)
            else:
                predicted_probs = unida_model(images)       
    else:
        if tta_method in ["stamp", "unida_tta"]:
            # reloading images
            pil_imgs = []
            for idx in img_indices.tolist():
                pil, _ = unida_dataset.load_img(idx)  # raw PIL before normalization
                pil_imgs.append(pil)
            
            # applying TTA transformations
            tta_transformations = get_tta_transforms(dataset_key)
            augmented_batches = []
            for _ in range(tta_num_augmentations):
                batch_tensors = []
                for pil in pil_imgs:
                    pil_for_aug = unida_dataset.test_transform(pil)
                    t_aug = tta_transformations(pil_for_aug)
                    batch_tensors.append(t_aug)             

                # stack into a [B, C, H, W] tensor and move to device
                aug_batch = torch.stack(batch_tensors, dim=0).to(unida_model.device)
                augmented_batches.append(aug_batch)
            
            # obtaining predictions
            images_and_aug_tensors = [images] + augmented_batches
            if model_key == "tasc":
                predicted_probs, unknown_scores_dict = unida_model(images_and_aug_tensors)
            else:
                predicted_probs = unida_model(images_and_aug_tensors)
        
        elif tta_method in ["owttt", "tent", "sotta", "sotta_v2", "eata", "gmm"]:
            logger.info(f"images.shape = {images.shape} (should be [B, C, H, W])")
            with torch.no_grad():
                if model_key == "tasc":
                    predicted_probs, unknown_scores_dict = unida_model(images)
                else:
                    predicted_probs = unida_model(images)
        elif "oracle" in tta_method:
            logger.info(f"images.shape = {images.shape} (should be [B, C, H, W])")
            with torch.no_grad():
                if model_key == "tasc":
                    predicted_probs, unknown_scores_dict = unida_model(images, gt_labels)
                else:
                    predicted_probs = unida_model(images, gt_labels)
        else:
            raise ValueError(f"Unknown TTA method: {tta_method}")

    logging.info(f"predicted probabilities obtained")
    logging.info(f"[DEBUG] predicted_probs.shape = {predicted_probs.shape} (should be [B, num_classes])")

    # print the top‐1 class for the first two images in this batch
    top1 = predicted_probs.argmax(dim=1)               # shape [B]
    logging.info(f"[DEBUG] top1 predicted classes (first 2): {top1[:2].tolist()}")
    logging.info(f"[DEBUG] GT labels (first 2): {gt_labels[:2].tolist()}")

    # storing predictions and ground-truths
    pred_probs_stack.append(predicted_probs.cpu())
    gt_label_stack.append(gt_labels)

    # stacking all predictions and ground-truths
    pred_cls_all = torch.cat(pred_probs_stack, dim=0)
    gt_label_all = torch.cat(gt_label_stack, dim=0)

    # calculating metrics
    if model_key == "tasc":
        # storing unknown scores
        unknown_scores_dict_stack["MS-s"].append(unknown_scores_dict["MS-s"].cpu())
        unknown_scores_dict_stack["MS-t"].append(unknown_scores_dict["MS-t"].cpu())
        unknown_scores_dict_stack["MS-s-w/ent"].append(unknown_scores_dict["MS-s-w/ent"].cpu())
        unknown_scores_dict_stack["MS-t-w/ent"].append(unknown_scores_dict["MS-t-w/ent"].cpu())
        unknown_scores_dict_stack["UniMS"].append(unknown_scores_dict["UniMS"].cpu())

        # stacking all unknown scores into a single big batch
        unknown_scores_dict_all = {
            "MS-s": torch.cat(unknown_scores_dict_stack["MS-s"], dim=0),
            "MS-t": torch.cat(unknown_scores_dict_stack["MS-t"], dim=0),
            "MS-s-w/ent": torch.cat(unknown_scores_dict_stack["MS-s-w/ent"], dim=0),
            "MS-t-w/ent": torch.cat(unknown_scores_dict_stack["MS-t-w/ent"], dim=0),
            "UniMS": torch.cat(unknown_scores_dict_stack["UniMS"], dim=0)
        } 

        h_score, known_acc, unknown_acc, per_class_acc, updated_labels = h_score_and_accuracies_tasc(unida_model, 
                                                                                        target_class_list,
                                                                                        gt_label_all,
                                                                                        pred_cls_all,
                                                                                        unknown_scores_dict_all,
                                                                                        open_flag,
                                                                                        inference=True)
    else:
        h_score, known_acc, unknown_acc, per_class_acc, updated_labels = h_score_and_accuracies_sfunida(target_class_list,
                                                                                        gt_label_all,
                                                                                        pred_cls_all,
                                                                                        open_flag)

    return h_score, known_acc, unknown_acc, per_class_acc, updated_labels

def get_nouns_from_wordnet(path=None):
    """
    Obtains nouns from WordNet
    """

    if not os.path.exists(path):
        # extract nouns from WordNet
        nouns = list(wordnet.all_synsets(pos="n"))
        nouns = [noun.name().split(".")[0] for noun in nouns]
        nouns = list(Counter([noun for noun in nouns]))  # use Counter to eliminate redundancy
        nouns = [noun.replace("_", " ") for noun in nouns]  # nouns in which "_" has been replaced by " "

        # save to pickle
        with open(path, "wb") as f:
            pkl.dump(np.char.array(nouns), f)
            
        return np.char.array(nouns)

    else:
        # load from pickle
        with open(path, "rb") as f:
            nouns_np = pkl.load(f)
        return nouns_np

def get_noun_embeddings(nouns_path, unida_model, embeddings_path, templates_type="ensemble", templates_sampling_rate=1):
    """
    Obtains CLIP embeddings for WordNet nouns using specified UniDA model
    """

    if not os.path.exists(embeddings_path):
        nouns_np = get_nouns_from_wordnet(nouns_path)
        noun_embeddings = unida_model.embed_classnames(classnames=nouns_np, templates_type=templates_type, 
                                                       templates_sampling_rate=templates_sampling_rate)
        with open(embeddings_path, "wb") as f:
            pkl.dump(noun_embeddings, f)
        return noun_embeddings
    else:
        with open(embeddings_path, "rb") as f:
            noun_embeddings = pkl.load(f)
        return noun_embeddings

def get_noun_similarities(input_imgs, noun_embeddings, unida_model, ce_prob_temp=0.02, topk=10):
    """
    Obtains similarities between image embeddings and noun embeddings
    
    Note: embeddings should already be normalized if using 
       - `embed_classnames` for nouns
       - `encode_image` for images
    """

    # obtaining img features
    input_imgs = input_imgs if input_imgs.ndim == 4 else input_imgs.unsqueeze(0)
    feat = unida_model.encode_image(input_imgs)

    # logits and probs 
    logits = feat @ noun_embeddings.T / ce_prob_temp
    probs = F.softmax(logits, dim=1)

    topk_probs, topk_indices = torch.topk(probs, k=topk, dim=1)

    return topk_probs, topk_indices

def get_nouns_from_index(noun_indices, nouns_path):
    """
    Obtains nouns from WordNet given their indices
    """

    nouns = get_nouns_from_wordnet(nouns_path).tolist()
    selected_nouns = [nouns[i] for i in noun_indices]
    return selected_nouns

if __name__ == "__main__":
    print("Module with functions/classes needed to run main pipeline functions")