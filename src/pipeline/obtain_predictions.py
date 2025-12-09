from omegaconf import DictConfig, OmegaConf

OUTPUT_DIR = "./output"

def evaluate_unida_tta_models(config: DictConfig):
    """
    Run Evaluation Pipeline for UniDA and TTA models
    """

    import numpy as np
    import logging, pickle, random
    from tqdm import tqdm
    from src.datasets.datasets import read_datalist, UnidaDataset
    import os, torch
    from torch.utils.data import DataLoader
    from src.pipeline.helpers import seed_worker, initializing_unida_tta_model, calculate_unida_tta_metrics, load_tta_config, wandb_assign_project_name
    from src.utils import obtain_all_transfers, obtain_config_transfer, obtain_config_dataset, setup_logging
    import wandb

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

    # setting up generator for dataloader
    g_target = torch.Generator()
    g_target.manual_seed(config.tta_runs.seed)

    # create output folder if needed
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info("output folder created at " + OUTPUT_DIR)

    logging.info("starting evaluation pipeline")
    logging.info(f"config: {config}")

    # setting global parameters for unida models
    model_key    = config.unida_runs.models.lower()      # e.g. LEAD or TASC
    dataset_key  = config.unida_runs.datasets.lower()     # e.g. Office or OfficeHome
    setting_key  = config.unida_runs.settings.lower()     # e.g. PDA, OPDA or OSDA
    transfer_key = config.unida_runs.transfers.lower()    # e.g. all or a specific transfer (use LEAD syntax)
    corruption   = config.unida_runs.corruption     # e.g. False or True
    tta_enabled  = config.tta_runs.tta_enabled     # e.g. False or True

    # corruption placeholders
    severity = None

    # tta placeholders
    tta_method = None
    tta_config = None
    tta_params = None
    tta_num_augmentations = None
    tta_grid_search = False
    consistency_filtering = None
    memory_reset = None
    risk_memory_size = None
    type_cf_booster = None # e.g. "balanced" or "unbalanced"
    use_unk_classifier = None
    
    # setting up corruption and tta parameters
    if corruption and tta_enabled:
        tta_grid_search = config.tta_runs.tta_grid_search # e.g. False or True
        corruption_type = config.unida_runs.corruption_type
        severity = config.unida_runs.corruption_severity # e.g. 1, 2, 3, 4, 5
        tta_method = config.tta_runs.tta_method 
        use_unk_classifier = config.tta_runs.use_unk_classifier # e.g. False or True
        
        if tta_method == "stamp":
            tta_num_augmentations = config.tta_runs.num_augmentations # e.g. 15
            consistency_filtering = config.tta_runs.consistency_filtering # e.g. False or True
            memory_reset = None # e.g. False or True
            risk_memory_size = None # e.g. 64, 128
        else:
            tta_num_augmentations = None
            consistency_filtering = None
            type_cf_booster = None
            memory_reset = None
            risk_memory_size = None
        
    elif corruption and not tta_enabled:
        corruption_type = config.unida_runs.corruption_type
        severity = config.unida_runs.corruption_severity 

    else:
        corruption_type = "none"   

    # setting up logging
    try:
        # build unique log filename using python f-strings
        log_filename = f"unida_tta_m-{model_key}_d-{dataset_key}_s-{setting_key}_t-{transfer_key}_c-{corruption}_tta-{tta_method}"
        log_filepath = os.path.join(OUTPUT_DIR, log_filename + ".log")

        setup_logging(log_filepath)

        current_node = os.environ["SLURM_JOB_NODELIST"]
        logging.info(f"current node: {str(current_node)}")

        if corruption:
            log_str = "evaluate model " + str(model_key) + " on dataset " + str(dataset_key) +\
                        " setting " + str(setting_key) + " transfers " + str(transfer_key) +\
                        " corruption " + str(corruption) + " corruption_type " + str(config.unida_runs.corruption_type) +\
                        " severity " + str(config.unida_runs.corruption_severity) + " tta_enabled " + str(tta_enabled)
        else:
            log_str = "evaluate model " + str(model_key) + " on dataset " + str(dataset_key) +\
                        " setting " + str(setting_key) + " transfers " + str(transfer_key) +\
                        " corruption " + str(corruption) + " tta_enabled " + str(tta_enabled)  
        logging.info(log_str)

    except Exception:
        logging.info("Failed at reading parametetrs")
        raise

    try: # gpu check
        if torch.cuda.is_available():
            logging.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("GPU is not available")

    except ImportError:
        logging.info("PyTorch not installed, cannot check GPU")
    
    # setting up wandb if enabled
    if config.logging.wandb: 

        project_name, tta_grid_search_params = wandb_assign_project_name(config, corruption, corruption_type, severity, tta_enabled, tta_method, 
                                                                         memory_reset, risk_memory_size, tta_grid_search, dataset_key, 
                                                                         model_key, use_unk_classifier)

        wandb.init(
            project=project_name,
            config=OmegaConf.to_container(config, resolve=True),
            mode="online"
        )

        # wandb table 1: final accuracy per class
        wandb_acc_table_cols = ["num_observations", "model", "setting", "dataset", "transfer", "corruption", "tta", "class", "accuracy"]
        wandb_acc_table = wandb.Table(columns=wandb_acc_table_cols)

        # wandb table 2: final overall metrics
        wandb_final_metrics_cols = ["num_observations", "model", "setting", "dataset", "transfer", "corruption", "tta", "avg_known_acc", "unknown_acc", "h_score", "tta_parameters"]
        wandb_final_metrics = wandb.Table(columns=wandb_final_metrics_cols)
        
        if not config.logging.get("wandb_run_name", None):
            try:
                if tta_enabled and tta_grid_search:
                    wandb.run.name = f"M: {model_key} | D: {dataset_key} | T: {transfer_key} | S: {setting_key} | C: {corruption} | CT: {corruption_type} | TTA Method: {tta_method} | TTA GS: {tta_grid_search_params.replace('.', '')}"
                elif tta_enabled and not tta_grid_search:
                    wandb.run.name = f"M: {model_key} | D: {dataset_key} | T: {transfer_key} | S: {setting_key} | C: {corruption} | CT: {corruption_type} | TTA: {tta_method}"
                else:
                    wandb.run.name = f"M: {model_key} | D: {dataset_key} | T: {transfer_key} | S: {setting_key} | C: {corruption} | CT: {corruption_type} | TTA: {tta_enabled}"
            except Exception:
                pass

    # setting up transfer parameters
    try:
        if transfer_key == "all":
            transfers = obtain_all_transfers(dataset_key)
        else:
            transfers = [transfer_key]
        logging.info(f"transfers: {str(transfers)}")
    except Exception:
        logging.info("Failed to obtain transfers")
        raise

    # calculating metrics for all transfers specified
    # NOTE: the following lines should be run when in evaluation mode only
    for transfer in transfers:
        try:
            logging.info(f"current transfer: {str(transfer)}")
            session_is_eval = config.session_is_eval 
            transfer_config = obtain_config_transfer(dataset_key, transfer)

            source = transfer_config["source"]
            target = transfer_config["target"]
            domain_type = transfer_config["domain_type"]

            model_path = config.unida_runs.model_paths[model_key]
                
        except Exception:
            logging.info("problem obtaining transfer configuration")
            raise

        ### obtaining dataset config for current setting
        try:
            # target data paths
            if corruption:
                target_data_dir = os.path.join(config.data_paths.corruptions, f"severity_{str(config.unida_runs.corruption_severity)}", corruption_type, dataset_key, target)
                target_list_path = os.path.join(config.data_paths.corruptions, f"severity_{str(config.unida_runs.corruption_severity)}", corruption_type, dataset_key, target, config.data_paths.labels_file_name)
                dataset_config = obtain_config_dataset(dataset_key, setting_key, os.path.join(config.data_paths.corruptions, f"severity_{str(config.unida_runs.corruption_severity)}", corruption_type),
                                                       source, target)
                logging.info(f"target_data_dir: {target_data_dir}")
            else:
                target_data_dir = os.path.join(config.data_paths.root, dataset_key, target)
                target_list_path = os.path.join(config.data_paths.root, dataset_key, target, config.data_paths.labels_file_name)
                dataset_config = obtain_config_dataset(dataset_key, setting_key, config.data_paths.root, source, target) 
                logging.info(f"target_data_dir: {target_data_dir}")          
            
            # source data paths
            source_data_dir = os.path.join(config.data_paths.root, dataset_key, source)
            source_list_path = os.path.join(config.data_paths.root, dataset_key, source, config.data_paths.labels_file_name)
            logging.info(f"source_data_dir: {source_data_dir}")            

            # number of classes per domain
            shared_class_num = dataset_config["shared_class_num"]
            source_private_class_num = dataset_config["source_private_class_num"]
            # target_private_class_num = dataset_config["target_private_class_num"] # TODO: check if needed for training
            
            if setting_key in ["pda", "opda"]: # base model was trained for all available classes in source domain
                num_classes = shared_class_num + source_private_class_num
            else: # osda
                num_classes = shared_class_num

            target_class_list = dataset_config["target_class_list"]
            source_class_list = dataset_config["source_class_list"]

            if tta_enabled:
                tta_config, tta_params = load_tta_config(tta_method, grid_search_enabled=tta_grid_search, config=config, model_path=model_path, num_classes=num_classes, 
                                                         source_class_list=source_class_list, target_class_list=target_class_list)

        except Exception:
            logging.exception("Failed to obtain dataset config")
            raise

        ### loading dataset
        try:
            ## target dataset 
            target_data_list = read_datalist(target_list_path)
            logging.info(f"Length of target datalist : {len(target_data_list)}")
            logging.info("target datalist loaded")
            
            preload_flg = False
            unida_dataset = UnidaDataset( ### dataset
                domain_type=domain_type,
                dataset=dataset_key,
                data_dir=target_data_dir,
                data_list=target_data_list,
                shared_class_num=dataset_config["shared_class_num"],
                source_private_class_num=dataset_config["source_private_class_num"],
                target_private_class_num=dataset_config["target_private_class_num"],
                unida_setting=setting_key,
                preload_flg=preload_flg
            )
            logging.info("target dataset loaded")

            data_loader = DataLoader(unida_dataset, batch_size=config.test_batch_size, shuffle=True,
                                     worker_init_fn=seed_worker,
                                     generator=g_target)
            final_num_obs = len(data_loader.dataset)
            logging.info("data loader instantiated for target domain")

            ## source dataset
            source_data_list = read_datalist(source_list_path)
            logging.info(f"Length of source datalist : {len(source_data_list)}")
            logging.info("source datalist loaded")
            
            preload_flg = False
            source_dataset = UnidaDataset( ### dataset
                domain_type="source",
                dataset=dataset_key,
                data_dir=source_data_dir,
                data_list=source_data_list,
                shared_class_num=dataset_config["shared_class_num"],
                source_private_class_num=dataset_config["source_private_class_num"],
                target_private_class_num=dataset_config["target_private_class_num"],
                unida_setting=setting_key,
                preload_flg=preload_flg
            )
            logging.info("source dataset loaded")

            source_data_loader = DataLoader(source_dataset, batch_size=config.test_batch_size, shuffle=True)
            source_num_obs = len(source_data_loader.dataset)
            logging.info(f"data loader instantiated for source domain, source_num_obs: {source_num_obs}")

        except Exception:
            logging.exception("Failed while loading dataset")
            raise

        ### loading model
        try:
            unida_model = initializing_unida_tta_model(model_key, num_classes, dataset_config, dataset_key, source, target, 
                                                       setting_key, model_path, tta_method, tta_config, 
                                                       consistency_filtering=consistency_filtering,
                                                       type_cf_booster=type_cf_booster,
                                                       memory_reset=memory_reset,
                                                       risk_memory_size=risk_memory_size,
                                                       source_loader=source_data_loader)
            
        except Exception:
            logging.exception("Failed while initialising model")
            raise

        ### evaluation
        try:
            # wandb table params
            if setting_key in ["osda", "opda"]:
                open_flag = True
            else: # pda
                open_flag = False

            # placeholders for ground-truth and predicted labels
            pred_probs_stack = []
            gt_label_stack = []
            img_indices_stack = []
            unknown_scores_dict_stack = {"MS-s": [], "MS-t": [], "MS-s-w/ent": [], "MS-t-w/ent": [], "UniMS": []} # tasc only
            logging.info(f"obtained target class list (open_flag is {open_flag}): {target_class_list}")

            # obtaining predictions for each batch of images
            count = 1
            list_test_batches = []
            per_class_acc_dict = {}
            
            for images, gt_labels, img_indices in tqdm(data_loader, ncols=60):
                # current number of obs
                list_test_batches.append(count)
                logging.info(f"img_indices: {img_indices}")

                h_score, known_acc, unknown_acc, per_class_acc, updated_labels = calculate_unida_tta_metrics(images, model_key, tta_enabled, tta_method, 
                                                                                                             unida_model, pred_probs_stack, 
                                                                                                             unknown_scores_dict_stack, 
                                                                                                             gt_label_stack, target_class_list, gt_labels, 
                                                                                                             img_indices, unida_dataset, dataset_key, 
                                                                                                             open_flag, tta_num_augmentations)

                img_indices_stack.append(img_indices) # stacking img indices                                                                             

                # wandb: storing accuracies per class
                per_class_acc = per_class_acc.tolist()
                for class_acc_index in range(len(per_class_acc)):
                    # known classes
                    if f"Class {class_acc_index}" not in per_class_acc_dict.keys() and class_acc_index != len(per_class_acc) - 1:
                        per_class_acc_dict[f"Class {class_acc_index}"] = []
                    elif f"Class {class_acc_index}" in per_class_acc_dict.keys():
                        per_class_acc_dict[f"Class {class_acc_index}"].append(per_class_acc[class_acc_index])
                    # unknown class
                    elif "Unknown Class" not in per_class_acc_dict.keys() and class_acc_index == len(per_class_acc) - 1:
                        per_class_acc_dict["Unknown Class"] = []
                    elif "Unknown Class" in per_class_acc_dict.keys() and class_acc_index == len(per_class_acc) - 1:
                        per_class_acc_dict["Unknown Class"].append(per_class_acc[class_acc_index])

                if open_flag:
                    logging.info(f"batch: {count}, h_score: {h_score}, known_acc: {known_acc}, unknown_acc: {unknown_acc}")
                else:
                    logging.info(f"count: {count}, h_score: {h_score}, known_acc: {known_acc}")

                logging.info(f"iteration: {count}, h_score: {h_score}, known_acc: {known_acc}, unknown_acc: {unknown_acc}, per_class_acc: {per_class_acc}")
                count += 1

            if config.logging.wandb:
                if tta_enabled: # aux tta method var for wandb logging
                    tta = tta_method
                else:
                    tta = "disabled"

                for accuracy_index in range(len(per_class_acc)):
                    logging.info("storing final accuracy per class")
                    if accuracy_index == len(per_class_acc) - 1 and open_flag:
                        wandb_acc_table.add_data(final_num_obs, model_key, setting_key, dataset_key, transfer, corruption_type, tta, "Unknown", per_class_acc[accuracy_index])
                    else:
                        wandb_acc_table.add_data(final_num_obs, model_key, setting_key, dataset_key, transfer, corruption_type, tta, str(accuracy_index), per_class_acc[accuracy_index])
                    logging.info("stored final accuracy per class")
                        
                logging.info(f"storing final metrics in wandb")
                if tta_grid_search:
                    wandb_final_metrics.add_data(final_num_obs, model_key, setting_key, dataset_key, transfer, corruption_type, tta, known_acc, unknown_acc, h_score, tta_grid_search_params)
                else:
                    wandb_final_metrics.add_data(final_num_obs, model_key, setting_key, dataset_key, transfer, corruption_type, tta, known_acc, unknown_acc, h_score, tta_params)
                logging.info(f"stored final metrics in wandb")

            if config.save_predictions:
                logging.info("storing results in output folder")

                # creating output folder
                if corruption and tta_enabled and tta_grid_search:
                    predictions_output_path = os.path.join(config.predictions_path, dataset_key, "corruption_tta", tta_method, "gs", tta_grid_search_params, corruption_type, model_key, setting_key, transfer)
                elif corruption and tta_enabled and not tta_grid_search:
                    predictions_output_path = os.path.join(config.predictions_path, dataset_key, f"corruption_tta", tta_method, "no_gs", corruption_type, model_key, setting_key, transfer)
                elif corruption and not tta_enabled:
                    predictions_output_path = os.path.join(config.predictions_path, dataset_key, "corruption", corruption_type, model_key, setting_key, transfer)
                else:
                    predictions_output_path = os.path.join(config.predictions_path, dataset_key, "no_corruption", model_key, setting_key, transfer)
                os.makedirs(predictions_output_path, exist_ok=True)

                # storing predictions and ground-truths
                with open(os.path.join(predictions_output_path, "pred_probs_stack.pkl"), "wb") as handle:
                    pickle.dump(pred_probs_stack, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                with open(os.path.join(predictions_output_path, "gt_label_stack.pkl"), "wb") as handle:
                    pickle.dump(gt_label_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(predictions_output_path, "unknown_scores_dict_stack.pkl"), "wb") as handle:
                    pickle.dump(unknown_scores_dict_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                with open(os.path.join(predictions_output_path, "pred_labels.pkl"), "wb") as handle:
                    pickle.dump(updated_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)           
                with open(os.path.join(predictions_output_path, "img_indices_stack.pkl"), "wb") as handle:
                    pickle.dump(img_indices_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)            

                logging.info(f"stored predictions at {predictions_output_path}")

        except Exception:
            logging.exception("Failed while calculating metrics")
            raise

    try:
        if config.logging.wandb:
            wandb.log({
                        f"all_accuracies": wandb_acc_table,
                        f"all_metrics": wandb_final_metrics
                    })

            logging.info("wandb tables logged")
            wandb.finish()

        logging.info("Finished evaluation pipeline")
        sys.exit("Forcing exit after succesful run of evaluation pipeline")
    except Exception:
        logging.exception("Failed while logging metrics")
        raise

if __name__ == "__main__":
    print("Module with functions/classes needed to run main evaluation of unida and tta")