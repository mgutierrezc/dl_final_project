import itertools, json
import os, pickle, string, time, torch, wandb
import pandas as pd
import logging, sys, faulthandler, signal
from src.models.tta_models import setup_tta
from omegaconf import DictConfig, OmegaConf

DATASET_CONFIG = "config/dataset/dataset_config.json"

#### generic helpers ####

def create_pickle(object_name, file_name: str, path: str) -> None:
    """
    Creates a pickle file for object. Note: Path should have no slash 
    at the end
    """
    with open(path + f"/{file_name}", "wb") as storing_output:
        pickle.dump(object_name, storing_output)
        storing_output.close()

def read_pickle(file_name: str, path: str) -> None:
    """
    Reads pickle file from specified path 
    """
    pickle_file = open(path + f"/{file_name}", "rb")
    output = pickle.load(pickle_file)
    pickle_file.close()
    return output

def folder_creator(folder_name: string, path: string) -> None:
    """
    Generates a folder in specified path
    
    Input: name of root folder, path where you want 
    folder to be created
    Output: None
    """
    
    # defining paths
    data_folder_path = path + "/" + folder_name
    data_folder_exists = os.path.exists(data_folder_path)

    # creating folders if don't exist
    if data_folder_exists:
        pass
    else:    
        # create a new directory because it does not exist 
        os.makedirs(data_folder_path)

        # create subfolders
        print(f"The new directory '{folder_name}' was created successfully! \nYou can find it on the following path: {path}")

def log_timer(stop_event, timer=20):
    """
    Determines if a snippet is still running after 
    `timer` seconds
    """

    while not stop_event.is_set():
        logging.info("Running current process...")
        time.sleep(timer)

def read_txt_to_label_dict(txt_path: str):
    """
    Reads txt file with image paths and numeric labels, extracts text labels, 
    and returns mapping of numeric to text labels
    """

    logger = logging.getLogger(__name__)
    logging.info("started read_txt_to_label_dict")

    logging.info(f"Reading txt file {txt_path} to obtain class labels")

    df = pd.read_csv(txt_path, sep=" ", header=None, names=["image_path", "class_num_label"])
        
    if "office" in txt_path.lower() and "officehome" not in txt_path.lower():
        df["class_text_label"] = df["image_path"].str.extract(r'images/([^/]+)/')
    else:
        df["class_text_label"] = df["image_path"].apply(lambda image_path: image_path.split("/")[0])

    df = df.drop_duplicates(subset=["class_num_label", "class_text_label"])
    logging.info("completed read_txt_to_label_dict")
    return dict(zip(df["class_num_label"], df["class_text_label"]))

def setup_logging(log_path: str, level=logging.INFO):
    """
    Sets up the global logging for the scripts being execute
    """
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)

    # uncaught Python exceptions
    def _ex_hook(exc_type, exc, tb):
        root.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
    sys.excepthook = _ex_hook

    # native crashes
    raw_log = open(log_path, "a", buffering=1)   # line‑buffered, low‑level
    faulthandler.enable(file=raw_log, all_threads=True)

    # optional: manual dump on SIGUSR1
    try:
        faulthandler.register(signal.SIGUSR1, file=fh.stream, all_threads=True)
    except (AttributeError, ValueError):
        # SIGUSR1 may not be available or already in use
        pass

    # optional: log receipt of SIGTERM
    def _term_handler(signum, frame):
        root.error("Received SIGTERM, exiting")
        sys.exit(1)
    signal.signal(signal.SIGTERM, _term_handler)

def remap_state_dict(state_dict):
    """
    Remaps keys of layers found in state dictionary
    """

    new = {}

    for key_layer, value_weights in state_dict.items():
        if key_layer.startswith("backbone_layer."):
            new["backbone." + key_layer[len("backbone_layer."):]] = value_weights
        elif key_layer.startswith("feat_embed_layer."):
            new["embedding_layer." + key_layer[len("feat_embed_layer."):]] = value_weights
        elif key_layer.startswith("class_layer."):
            new["model." + key_layer[len("class_layer."):]] = value_weights
        else:                       # anything else, keep as is
            new[key_layer] = value_weights
    
    return new

def obtain_all_transfers(dataset_key, datasets_config_path="config/dataset/dataset_config.json"):
    """
    Returns a list with all transfer keys for a given dataset
    """

    # obtain all possible domain transfers
    with open(datasets_config_path, "r") as file:
        datasets_config = json.load(file)
        domain_list = datasets_config[dataset_key]["domain_list"]

    # choose abbreviation length
    if dataset_key.lower() != "officehome":
        # first letter of each domain
        domain_codes = [domain[0].lower() for domain in domain_list]
    else:
        # first two letters of each domain
        domain_codes = [domain[:2].lower() for domain in domain_list]

    # all 2-element permutations, joined by "2", sorted alphabetically
    all_transfer_keys = [f"{a}2{b}" for a, b in itertools.permutations(domain_codes, 2)]
    all_transfer_keys.sort()

    return all_transfer_keys

def obtain_config_transfer(dataset_key, transfer_key, session_is_eval=True,
                           datasets_config_path="config/dataset/dataset_config.json"):
    """
    Returns a list with all transfer-specific parameters
    """
    
    config = {}
    with open(datasets_config_path, "r") as file:
        dataset_config = json.load(file)[dataset_key]
        domain_list = dataset_config["domain_list"]
    
    # obtaining mapping of domain keys to domain names
    if dataset_key.lower() != "officehome":
        domain_map = {domain[0].lower(): domain for domain in domain_list}
    else:
        domain_map = {domain[:2].lower(): domain for domain in domain_list}

    source_key, target_key = transfer_key.split("2") # obtain source and target domains
    
    # obtaining source and target domains
    config["source"] = domain_map[source_key]
    config["target"] = domain_map[target_key]

    if session_is_eval:
        config["domain_type"] = "target"
    else:
        raise ValueError("Training configs not supported yet")
    
    return config

def obtain_checkpoint_path(model_key, dataset_key, source, target, setting_key, model_path):
    """
    Obtains path to the checkpoint of respective model
    """

    logger = logging.getLogger(__name__)
    logger.info("starting obtain_checkpoint_path")

    if model_key == "lead":
        if dataset_key == "office":
            lead_mapping = {
                            "amazon": "0",
                            "dslr": "1",
                            "webcam": "2"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_0.3/{dataset_key}_SFDA_best_target_checkpoint.pth")
        elif dataset_key == "officehome":
            lead_mapping = {
                            "art": "0",
                            "clipart": "1",
                            "product": "2",
                            "realworld": "3"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_2.0/{dataset_key}_SFDA_best_target_checkpoint.pth")
        elif dataset_key == "domainnet":
            lead_mapping = {
                            "painting": "0",
                            "real": "1",
                            "sketch": "2"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_2.0/{dataset_key}_SFDA_best_target_checkpoint.pth")
        elif dataset_key == "visda":
            lead_mapping = {
                            "train": "0",
                            "validation": "1"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_1.0/{dataset_key}_SFDA_best_target_checkpoint.pth")
        else:
            raise NotImplementedError(f"Unknown dataset specified: {dataset_key}")
    elif model_key == "tasc":
        tasc_mapping = {"office": 
                            {
                                "amazon": "a",
                                "dslr": "d",
                                "webcam": "w"
                            },
                        "officehome":
                            {
                                "art": "A",
                                "clipart": "C",
                                "product": "P",
                                "realworld": "R"
                            },
                        "domainnet":
                            {
                                "painting": "p",
                                "real": "r",
                                "sketch": "s"
                            },
                        "visda":
                            {
                                "train": "S",
                                "validation": "R"
                            }
                        }

        runs_dir_model_classnames = os.path.join(model_path + "_pth", dataset_key) 
        runs_dir_cluster_params = os.path.join(model_path, dataset_key)
        
        # finding model + classnames subfolder with the setting_key
        if setting_key == "osda":
            setting_key = "oda"
        subfolders_model_classnames = [name for name in os.listdir(runs_dir_model_classnames) if os.path.isdir(os.path.join(runs_dir_model_classnames, name)) and f"_{setting_key.upper()}_" in name]
        subfolders_cluster_params = [name for name in os.listdir(runs_dir_cluster_params) if os.path.isdir(os.path.join(runs_dir_cluster_params, name)) and f"_{setting_key.upper()}_" in name]        

        logger.info(f"setting_key: {setting_key.upper()}")
        logger.info(f"Subfolders found in {runs_dir_model_classnames}: {subfolders_model_classnames}")

        source_key = tasc_mapping[dataset_key][source]
        target_key = tasc_mapping[dataset_key][target]
        source_target = f"_{source_key}{target_key}_"

        for subfolder in subfolders_model_classnames:
            if source_target in subfolder:
                logger.info(f"Found model + classnames subfolder: {subfolder} for source {source} and target {target}")
                checkpoint_path = os.path.join(runs_dir_model_classnames, subfolder, "final_model.pth")
                target_classnames_gsearch = os.path.join(runs_dir_model_classnames, subfolder, "target_classnames.txt")
            else:
                pass
        
        for subfolder in subfolders_cluster_params:
            if source_target in subfolder:
                logger.info(f"Found cluster subfolder: {subfolder} for source {source} and target {target}")
                cluster_params_path = os.path.join(runs_dir_cluster_params, subfolder, "cluster_params.json")
            else:
                pass
        
        try:
            return checkpoint_path, target_classnames_gsearch, cluster_params_path
        except:
            # if no subfolder found, raise error
            raise FileNotFoundError(f"Checkpoint for {dataset_key} with source {source} and target {target} not found in {runs_dir_model_classnames}")
    else:
        raise NotImplementedError(f"Unknown model specified: {model_key}.")

def obtain_model_files_path(model_key, dataset_key, source, target, setting_key, model_path, files="source_features"):
    """
    Obtains path to the features of respective model
    """

    logger = logging.getLogger(__name__)
    logger.info("starting obtain_source_features_path")

    if model_key == "lead":
        if dataset_key == "office":
            lead_mapping = {
                            "amazon": "0",
                            "dslr": "1",
                            "webcam": "2"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_0.3/{dataset_key}_{files}")
        elif dataset_key == "officehome":
            lead_mapping = {
                            "art": "0",
                            "clipart": "1",
                            "product": "2",
                            "realworld": "3"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_2.0/{dataset_key}_{files}")
        elif dataset_key == "domainnet":
            lead_mapping = {
                            "painting": "0",
                            "real": "1",
                            "sketch": "2"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_2.0/{dataset_key}_{files}")
        elif dataset_key == "visda":
            lead_mapping = {
                            "train": "0",
                            "validation": "1"
                            }
            return os.path.join(model_path, 
                                f"{dataset_key}/s_{lead_mapping[source]}_t_{lead_mapping[target]}/{setting_key}/smooth_psd_1.0/{dataset_key}_{files}")
        else:
            raise NotImplementedError(f"Unknown dataset specified: {dataset_key}")
    elif model_key == "tasc":
        tasc_mapping = {"office": 
                            {
                                "amazon": "a",
                                "dslr": "d",
                                "webcam": "w"
                            },
                        "officehome":
                            {
                                "art": "A",
                                "clipart": "C",
                                "product": "P",
                                "realworld": "R"
                            },
                        "domainnet":
                            {
                                "painting": "p",
                                "real": "r",
                                "sketch": "s"
                            },
                        "visda":
                            {
                                "train": "S",
                                "validation": "R"
                            }
                        }

        runs_dir_model_classnames = os.path.join(model_path + "_pth", dataset_key) 
        
        # finding model + classnames subfolder with the setting_key
        if setting_key == "osda":
            setting_key = "oda"
        subfolders_model_classnames = [name for name in os.listdir(runs_dir_model_classnames) if os.path.isdir(os.path.join(runs_dir_model_classnames, name)) and f"_{setting_key.upper()}_" in name]
        
        logger.info(f"setting_key: {setting_key.upper()}")
        logger.info(f"Subfolders found in {runs_dir_model_classnames}: {subfolders_model_classnames}")

        source_key = tasc_mapping[dataset_key][source]
        target_key = tasc_mapping[dataset_key][target]
        source_target = f"_{source_key}{target_key}_"

        for subfolder in subfolders_model_classnames:
            if source_target in subfolder:
                logger.info(f"Found model + classnames subfolder: {subfolder} for source {source} and target {target}")
                checkpoint_path = os.path.join(runs_dir_model_classnames, subfolder, files)
            else:
                pass
                
        try:
            return checkpoint_path
        except:
            # if no subfolder found, raise error
            raise FileNotFoundError(f"Checkpoint for {dataset_key} with source {source} and target {target} not found in {runs_dir_model_classnames}")
    else:
        raise NotImplementedError(f"Unknown model specified: {model_key}.")


def obtain_config_dataset(dataset_key, unida_setting, dataset_path="./data", source="amazon", 
                          target="dslr"):
    """
    Returns a dictionary with all dataset-specific parameters
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Obtaining config for dataset {dataset_key} with setting {unida_setting} from {dataset_path}")

    config = {}
    with open(DATASET_CONFIG, "r") as file:
        datasets_config = json.load(file)

    # directories of domains
    source_data_dir = os.path.join(dataset_path, dataset_key, source)
    target_data_dir = os.path.join(dataset_path, dataset_key, target)
    
    # number of classes per split
    shared_class_num = datasets_config[dataset_key][unida_setting]["shared_class_num"]
    source_private_class_num = datasets_config[dataset_key][unida_setting]["source_private_class_num"]
    target_private_class_num = datasets_config[dataset_key][unida_setting]["target_private_class_num"]

    embed_feat_dim = datasets_config[dataset_key] # num of features for dataset

    # number of classes 
    source_class_num = shared_class_num + source_private_class_num
    target_class_num = shared_class_num + target_private_class_num

    # class numeric labels per domain (source)
    source_class_list = [i for i in range(source_class_num)]
    target_class_list = [i for i in range(shared_class_num)]

    # class text labels per domain (target)
    source_labels_mapping = read_txt_to_label_dict(os.path.join(source_data_dir, "image_unida_list.txt"))
    source_classnames = [source_labels_mapping[i] for i in source_class_list]
    logger.info(f"Source classnames: {source_classnames}")

    target_labels_mapping = read_txt_to_label_dict(os.path.join(target_data_dir, "image_unida_list.txt"))
    target_classnames = [target_labels_mapping[i] for i in target_class_list]

    if target_private_class_num > 0: # adding label for unknown classes (OSDA/OPDA)
        target_class_list.append(source_class_num)

    config.update({ # updating config with domain variables
        "dataset": dataset_key,
        "embed_feat_dim": embed_feat_dim,
        "source_data_dir": source_data_dir,
        "target_data_dir": target_data_dir,
        "shared_class_num": shared_class_num,
        "source_private_class_num": source_private_class_num,
        "target_private_class_num": target_private_class_num,
        "source_class_num": source_class_num,
        "target_class_num": target_class_num,
        "class_num": source_class_num,
        "source_class_list": source_class_list,
        "target_class_list": target_class_list,
        "source_classnames": source_classnames,
        "target_classnames": target_classnames
    })

    logger.info(f"Config for dataset {dataset_key} with setting {unida_setting} obtained successfully.")
    return config

#### pipeline helpers ####

def obtain_transfers_params(config: DictConfig):
    """
    Obtains all transfer learning settings to evaluate based on the config
    """

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
            # type_cf_booster = config.tta_runs.type_cf_booster # e.g. "balanced" or "unbalanced"
            # memory_reset = config.tta_runs.memory_reset # e.g. False or True
            # risk_memory_size = config.tta_runs.risk_memory_size # e.g. 64, 128
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

        project_name, tta_grid_search_params = wandb_assign_project_name(config, corruption_type, severity, tta_enabled, tta_method, 
                                  consistency_filtering, type_cf_booster, memory_reset, risk_memory_size, tta_grid_search, dataset_key, 
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

    return trasfers_config, transfers