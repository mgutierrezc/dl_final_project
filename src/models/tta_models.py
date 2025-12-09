import pytorch_lightning as L
import logging, math, os, random, time
from copy import deepcopy
from .common_resources import copy_model_and_optimizer, load_model_and_optimizer
from torch.cuda.amp import autocast
from .unida_models import load_unida_model
from src.metrics import entropy
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
from src.metrics import np_threshold, np_optimal_classifier, vectorize_density, compute_os_variance
from src.utils import obtain_model_files_path, obtain_config_dataset
from functools import wraps
from src.datasets.datasets import get_tta_transforms, get_tta_transforms_gmm, UnidaDataset, read_datalist
from scipy.stats import multivariate_normal
from src.models.vit_resources.LoRA_layer import LORAMultiheadAttention

# CODE CITATIONS:
# the classes for the TTA come from https://github.com/yuyongcan/STAMP/tree/master
# the only class that has a separate source is GMM and comes from https://github.com/pascalschlachter/GMM/blob/main/adaptation.py

#### helpers ####

def setup_oracle(base_model, cfg, backbone_source=None, target_classnames_gsearch=None, cluster_params_path=None, source_classnames=None):
    
    logger = logging.getLogger(__name__)
    logger.info(f"setup_oracle with tta_config: {cfg}")
      
    model_key = cfg["current_eval"]["model_key"]
    num_classes = cfg["current_eval"]["num_classes"]
    source_class_list = cfg["current_eval"]["source_class_list"]
    target_class_list = cfg["current_eval"]["target_class_list"]

    type_of_oracle = next((k for k in cfg if "oracle" in k), None)

    cfg_params_oracle = cfg[type_of_oracle]["params_oracle"]
    base_model = Oracle.configure_model(base_model, cfg_params_oracle)
    params, _ = Oracle.collect_params(base_model, cfg_params_oracle)

    logging.info(f"type_of_oracle: {type_of_oracle}")
    neg_mem_weight = cfg[type_of_oracle]["neg_mem_weight"]
    kl_weight = cfg[type_of_oracle]["kl_weight"]
    optimizer = optim.Adam(params,
                         lr=cfg["optim"]["lr"],
                         betas=(cfg["optim"]["beta"], 0.999),
                         weight_decay=cfg["optim"]["wd"])
    model = Oracle(base_model, optimizer, model_key, source_class_list=source_class_list, target_class_list=target_class_list, steps=1, backbone_source=backbone_source, 
                   target_classnames_gsearch=target_classnames_gsearch, cluster_params_path=cluster_params_path, source_classnames=source_classnames, 
                   neg_mem_weight=neg_mem_weight, kl_weight=kl_weight)

    return model

def setup_tent(base_model, cfg):

    logger = logging.getLogger(__name__)
    logger.info(f"setup_tent with tta_config: {cfg}")
    
    base_model = TENT.configure_model(base_model)
    params, _ = TENT.collect_params(base_model)
    model_key = cfg["current_eval"]["model_key"]
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]
    optimizer = optim.Adam(params,
                         lr=cfg["optim"]["lr"],
                         betas=(cfg["optim"]["beta"], 0.999),
                         weight_decay=cfg["optim"]["wd"])
    model = TENT(base_model, optimizer, model_key, use_unk_classifier=use_unk_classifier) 
    return model

def setup_sotta(base_model, cfg):

    base_model = SoTTA.configure_model(base_model)
    logger = logging.getLogger(__name__)
    logger.info(f"setup_sotta with tta_config: {cfg}")
    
    params, _ = SoTTA.collect_params(base_model)
    model_key = cfg["current_eval"]["model_key"]
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]
    base_optimizer = optim.Adam
    optimizer = SAM(params, base_optimizer, lr=cfg["optim"]["lr"], rho=cfg["optim"]["rho"], betas=(cfg["optim"]["beta"], 0.999), weight_decay=cfg["optim"]["wd"])
    model = SoTTA(base_model, optimizer, model_key, cfg["sotta"]["threshold"], use_unk_classifier=use_unk_classifier)
    return model

def setup_sotta_v2(base_model, cfg):

    base_model = SoTTA.configure_model(base_model)
    logger = logging.getLogger(__name__)
    logger.info(f"setup_sotta with tta_config: {cfg}")
    
    params, _ = SoTTA.collect_params(base_model)
    model_key = cfg["current_eval"]["model_key"]
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]
    base_optimizer = optim.Adam
    optimizer = SAM(params, base_optimizer, lr=cfg["optim"]["lr"], rho=cfg["optim"]["rho"], betas=(cfg["optim"]["beta"], 0.999), weight_decay=cfg["optim"]["wd"])
    model = SoTTA_v2(base_model, optimizer, model_key, cfg["sotta_v2"]["threshold"], use_unk_classifier=use_unk_classifier, neg_mem_weight=cfg["sotta_v2"]["neg_mem_weight"])
    return model

def setup_eata(base_model, cfg):

    base_model = EATA.configure_model(base_model)
    logger = logging.getLogger(__name__)
    logger.info(f"setup_eata with tta_config: {cfg}")
    
    # compute fisher informatrix
    batch_size_target = cfg["current_eval"]["batch_size"]

    #### parameters for fisher computation ####

    model_key = cfg["current_eval"]["model_key"]
    model_path = cfg["current_eval"]["model_path"]
    dataset_key = cfg["current_eval"]["dataset_key"]
    source = cfg["current_eval"]["source"]
    target = cfg["current_eval"]["target"]
    setting_key = cfg["current_eval"]["setting_key"]
    corruptions_path = cfg["current_eval"]["corruptions_path"]
    labels_file_name = cfg["current_eval"]["labels_file_name"]
    num_classes = cfg["current_eval"]["num_classes"]   
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]

    fishers_path = obtain_model_files_path(model_key, dataset_key, source, target, setting_key, model_path, files="fishers")
    fisher_matrices_file = os.path.join(fishers_path, "fisher_matrices.pkl")

    # obtaining fisher matrices
    if os.path.exists(fisher_matrices_file):
        logging.info(f"loading fisher matrices from {fisher_matrices_file}")
        with open(fisher_matrices_file, 'rb') as f:
            fishers = torch.load(f)
        logging.info("fisher matrices loaded")
    else:
        logging.info("fisher matrices not found, computing fisher matrices...")
        target_data_dir = os.path.join(corruptions_path, "severity_5", "gaussian_noise", dataset_key, target)
        target_list_path = os.path.join(corruptions_path, "severity_5", "gaussian_noise", dataset_key, target, labels_file_name)
        dataset_config = obtain_config_dataset(dataset_key, setting_key, os.path.join(corruptions_path, "severity_5", "gaussian_noise"),
                                                source, target)
        
        ## target dataset 
        target_data_list = read_datalist(target_list_path)
        logging.info(f"Length of target datalist : {len(target_data_list)}")
        logging.info("target datalist loaded")
        
        fisher_dataset = UnidaDataset( ### dataset
            domain_type="target",
            dataset=dataset_key,
            data_dir=target_data_dir,
            data_list=target_data_list,
            shared_class_num=dataset_config["shared_class_num"],
            source_private_class_num=dataset_config["source_private_class_num"],
            target_private_class_num=dataset_config["target_private_class_num"],
            unida_setting=setting_key,
            preload_flg=False
        )

        logging.info("fisher dataset loaded")
        logging.info(f"target_data_dir: {target_data_dir}")

        # NOTE: not all UniDA datasets have large enough data for fisher, so we'll limit its calculations to min(500, length of the gaussian noise dataset)
        # NOTE: 500 comes from EATA ImageNet experiments
        subset_size = min(500, len(fisher_dataset))
        subset_indices = random.sample(range(len(fisher_dataset)), subset_size)
        fisher_dataset = torch.utils.data.Subset(fisher_dataset, subset_indices)
        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=batch_size_target, shuffle=True, pin_memory=True)

        model = EATA.configure_model(base_model)
        params, param_names = EATA.collect_params(base_model)
        ewc_optimizer = optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, batch in enumerate(fisher_loader, start=1):
            images = batch[0].cuda(non_blocking=True)

            if model_key == "tasc":
                outputs, _ = model(images)
            else:
                outputs = model(images)

            _, targets = outputs.max(1) # targets represents indices
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logging.info("compute fisher matrices finished")
        del ewc_optimizer
        del model

        logging.info(f"storing fisher matrices to {fisher_matrices_file}")
        os.makedirs(fishers_path, exist_ok=True)
        with open(fisher_matrices_file, "wb") as f:
            torch.save(fishers, f)

    params, _ = EATA.collect_params(base_model)
    optimizer = optim.Adam(params,
                         lr=cfg["optim"]["lr"],
                         betas=(cfg["optim"]["beta"], 0.999),
                         weight_decay=cfg["optim"]["wd"])
    eta_model = EATA(base_model, optimizer,
                     adapted_model_key=cfg["current_eval"]["model_key"],
                     steps=1,
                     fishers=fishers,
                     fisher_alpha=cfg["eata"]["fisher_alpha"],
                     e_margin=math.log(num_classes) * (0.4),
                     d_margin=cfg["eata"]["d_margin"], 
                     use_unk_classifier=use_unk_classifier
                     )

    return eta_model

def setup_stamp(base_model, cfg, model_key, checkpoint_path, backbone_source,
                target_classnames_gsearch=None, cluster_params_path=None,
                source_classnames=None, consistency_filtering=None):

    logger = logging.getLogger(__name__)
    logger.info(f"setup_stamp with tta_config: {cfg}")
    
    params, _ = STAMP.collect_params(base_model)
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]    
    base_optimizer = optim.SGD # base optimizer for SAM
    optimizer = SAM(params, base_optimizer, lr=cfg["optim"]["lr"], rho=cfg["optim"]["rho"])
    if model_key == "tasc":
        model = STAMP(base_model, model_key, checkpoint_path, backbone_source, optimizer, cfg["stamp"]["alpha"],
                      target_classnames_gsearch=target_classnames_gsearch,
                      cluster_params_path=cluster_params_path,
                      source_classnames=source_classnames, consistency_filtering=consistency_filtering,
                      use_unk_classifier=use_unk_classifier)
    else:
        model = STAMP(base_model, model_key, checkpoint_path, backbone_source, optimizer, cfg["stamp"]["alpha"],
                      consistency_filtering=consistency_filtering, use_unk_classifier=use_unk_classifier)
    return model

def setup_gmm(base_model, cfg):
    logger = logging.getLogger(__name__)
    logger.info(f"setup_gmm with tta_config: {cfg}")

    dataset_key = cfg["current_eval"]["dataset_key"]
    setting_key = cfg["current_eval"]["setting_key"]
    corruptions_path = cfg["current_eval"]["corruptions_path"]
    source = cfg["current_eval"]["source"]
    target = cfg["current_eval"]["target"]
    use_unk_classifier = cfg["current_eval"]["use_unk_classifier"]    

    dataset_config = obtain_config_dataset(dataset_key, setting_key, os.path.join(corruptions_path, "severity_5", "gaussian_noise"),
                                           source, target)
    base_model = GmmBaAdaptationModule(base_model, model_name=cfg["current_eval"]["model_key"],
                                 shared_class_num=dataset_config["shared_class_num"],
                                 source_private_class_num=dataset_config["source_private_class_num"],
                                 lr=cfg["optim"]["lr"],
                                 red_feature_dim=int(cfg["gmm"]["red_feature_dim"]),
                                 p_reject=cfg["gmm"]["p_reject"],
                                 N_init=cfg["gmm"]["N_init"],
                                 augmentation=cfg["gmm"]["augmentation"],
                                 lam=cfg["gmm"]["lam"],
                                 temperature=cfg["gmm"]["temperature"], 
                                 use_unk_classifier=use_unk_classifier)
    try:
        base_model.to("cuda")
    except:
        raise ValueError("cuda not available")
    return base_model
    
def setup_tta(base_model, tta_method, cfg, model_key, checkpoint_path, backbone_source="pth",
              **kwargs):
    """
    Set up the model for test-time adaptation based on the configuration
    and specified adaptation method

    cfg comes from the global tta config
    base_model is the unida_model object
    tta_method is a string indicating the adaptation method to use
    """
    
    # TODO: update for gmm

    # setting up logger
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up test-time adaptation method: {tta_method}")

    if "oracle" in tta_method:
        model = setup_oracle(base_model, cfg, backbone_source=backbone_source,
                             target_classnames_gsearch=kwargs.get("target_classnames_gsearch", None),
                             cluster_params_path=kwargs.get("cluster_params_path", None),
                             source_classnames=kwargs.get("source_classnames", None))
    elif tta_method == "tent":
        model = setup_tent(base_model, cfg)
    elif tta_method == "eata":
        model = setup_eata(base_model, cfg)
    elif tta_method == "sotta":
        model = setup_sotta(base_model, cfg)
    elif tta_method == "sotta_v2":
        model = setup_sotta_v2(base_model, cfg)
    # elif tta_method == 'odin':
    #     model = setup_odin(base_model, cfg)
    elif tta_method == "stamp":
        model = setup_stamp(base_model, cfg, model_key, checkpoint_path, backbone_source,
                                target_classnames_gsearch=kwargs.get("target_classnames_gsearch", None),
                                cluster_params_path=kwargs.get("cluster_params_path", None),
                                source_classnames=kwargs.get("source_classnames", None),
                                consistency_filtering=kwargs.get("consistency_filtering", None))
    elif tta_method == "gmm":
        model = setup_gmm(base_model, cfg)
    else:
        raise ValueError(f"Adaptation method '{tta_method}' is not supported!")

    return model

def get_source_features_labels(model, source_loader, ckpt_dir=None):
    """
    Get the features of the source dataset.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"get_source_features_labels: source_loader {source_loader}")
    
    if ckpt_dir is not None: # reading features from checkpoint if available
        os.makedirs(ckpt_dir, exist_ok=True)
        features_path = os.path.join(ckpt_dir, "source_features.pt")
        labels_path = os.path.join(ckpt_dir, "source_labels.pt")
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features = torch.load(features_path)
            labels = torch.load(labels_path)
            logger.info("getting source feature from ckpt")
            return features, labels

        else: # obtaining features if no checkpoint available
            logger.info("getting source features")

            model.eval()
            features_all = []
            labels_all = []
            with torch.no_grad():
                for batch_idx, (inputs, targets, _) in enumerate(source_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with torch.no_grad():
                        features, logits = model(inputs, return_feats=True)
                        features_all.append(features)
                        labels_all.append(targets)

            features_all = torch.cat(features_all, dim=0)
            labels_all = torch.cat(labels_all, dim=0)

            logger.info(f"source feature shape: {features_all.shape}")
            torch.save(features_all, features_path)
            torch.save(labels_all, labels_path)
            
            return features_all, labels_all
    else:
        raise ValueError("ckpt_dir is not specified, cannot save source features and labels.")

def append_prototypes(pool, feat_ext, logit, ts, ts_pro):
    """
    Append strong OOD prototypes to the prototype pool
    """

    added_list = []
    update = 1

    while update:
        feat_mat = pool(F.normalize(feat_ext), all=True)
        if not feat_mat == None:
            new_logit = torch.cat([logit, feat_mat], 1)
        else:
            new_logit = logit

        r_i_pro, _ = new_logit.max(dim=-1)
        r_i, _ = logit.max(dim=-1)

        # if added_list is not empty, set the cosine similarity between the added features and the strong OOD 
        # prototypes to 1, to avoid the added features to be appended to the prototype pool again.
        if added_list != []:
            for add in added_list:
                r_i[add] = 1
        min_logit, min_index = r_i.min(dim=0)

        # if the cosine similarity between the feature and the weak OOD prototypes is less than the threshold ts, 
        # the feature is a strong OOD sample.
        if (1 - min_logit) > ts:
            
            added_list.append(min_index)
            if (1 - r_i_pro[min_index]) > ts_pro:
                # if this strong OOD sample is far away from all the strong OOD prototypes, append it to the prototype pool.
                pool.update_pool(F.normalize(feat_ext[min_index].unsqueeze(0)))
        else:
            # all the features are weak OOD samples, stop the loop.
            update = 0

def calculate_entropy_from_likelihood(likelihood):
    """
    Entropy function for likelihood (TTA: GMM)
    """

    entropy_values = -(likelihood * torch.log2(likelihood + 1e-10)).sum(dim=1)
    scale_factor = torch.log2(torch.tensor(likelihood.shape[1]))
    entropy_values = entropy_values / scale_factor
    return entropy_values

def calculate_cosine_similarity(mu, feat):
    cosine_sim = F.cosine_similarity(mu.unsqueeze(0), feat.unsqueeze(1), dim=2)
    return cosine_sim

def calculate_kld(likelihood, true_dist):
    T = 0.1
    dividend = torch.sum(torch.exp(likelihood / T), dim=1)
    logarithmus = - torch.log(dividend)
    divisor = torch.sum(true_dist, dim=1)
    kld_values = - (1 / likelihood.shape[1]) * divisor * logarithmus
    return kld_values

#### tta classes for models ####

class BaseModule(L.LightningModule):
    def __init__(self, model, model_name, shared_class_num, source_private_class_num, lr):
        super(BaseModule, self).__init__()

        self.model = model
        self.model_name = model_name
        self.feature_dim = model.feature_dim

        # TODO: needs to be updated
        self.known_classes_num = shared_class_num + source_private_class_num

        # possible hyperparameters for TTA
        self.lr = lr

class HUS:
    """
    High-confidence Uniform-class Sampling (HUS) memory for TTA
    """

    def __init__(self, capacity, num_class, threshold=None):
        self.num_class = num_class
        self.data = [[[], [], []] for _ in
                     range(self.num_class)]  # feat, pseudo_cls, domain, conf
        self.counter = [0] * self.num_class
        self.capacity = capacity
        self.threshold = threshold

    def get_memory(self):
        data = []
        for x in self.data:
            data.extend(x[0])
        try:
            data = torch.stack(data)
            return data
        except:
            logging.warning(f"HUS memory is empty! data: {data}")
            return None            # prevent torch.stack on empty

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 3)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.threshold is not None and instance[2] < self.threshold:
            is_add = False
        elif self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)
        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def get_target_index(self, data):
        return random.randrange(0, len(data))

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices:  # instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = self.get_target_index(self.data[largest][1])
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            tgt_idx = self.get_target_index(self.data[cls][1])
            for dim in self.data[cls]:
                dim.pop(tgt_idx)
        return True

    def reset_value(self, feats, cls, aux):
        self.data = [[[], [], []] for _ in range(self.num_class)]  # feat, pseudo_cls, domain, conf

        for i in range(len(feats)):
            tgt_idx = cls[i]
            self.data[tgt_idx][0].append(feats[i])
            self.data[tgt_idx][1].append(cls[i])
            self.data[tgt_idx][2].append(aux[i])

class RBM:
    """
    Reliable (Class) Balanced Memory or RBM for TTA
    """

    def __init__(self, max_len, num_class):
        self.num_classes = num_class
        self.count_class = torch.zeros(num_class)
        self.data = [[] for _ in range(num_class)]
        self.max_len = max_len
        self.total_num = 0

    def remove_item(self):
        max_count = 0
        for i in range(self.num_classes):
            if len(self.data[i]) == 0:
                continue
            if self.count_class[i] > max_count:
                max_count = self.count_class[i]
        max_classes = []
        for i in range(self.num_classes):
            if self.count_class[i] == max_count and len(self.data[i]) > 0:
                max_classes.append(i)
        remove_class = random.choice(max_classes)
        self.data[remove_class].pop(0)

    def append(self, items, class_ids):
        for item, class_id in zip(items, class_ids):
            if self.total_num < self.max_len:
                self.data[class_id].append(item)
                self.total_num += 1
            else:
                self.remove_item()
                self.data[class_id].append(item)

    def get_data(self):
        data = []
        for cls in range(self.num_classes):
            data.extend(self.data[cls])
            self.count_class[cls] = 0.9 * self.count_class[cls] + 0.1 * len(self.data[cls])
        return torch.stack(data)

    def __len__(self):
        return self.total_num

    def reset(self):
        self.count_class = torch.zeros(self.num_classes)
        self.data = [[] for _ in range(self.num_classes)]
        self.total_num = 0 
    
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class Prototype_Pool(nn.Module):
    """
    Prototype pool containing strong OOD prototypes for OWTTT
    """

    def __init__(self, max=100, memory=None):
        """
        Constructor method to initialize the prototype pool, storing the values of delta, the number of weak OOD categories,
        and the maximum count of strong OOD prototypes. 
        
        update_pool: Method to append and delete strong OOD prototypes.
        """

        super(Prototype_Pool, self).__init__()

        self.max_length = max
        self.flag = 0
        if memory is not None:
            self.register_buffer("memory", memory)
            self.flag = 1

    def forward(self, x, all=False):
        """
        Return the cosine similarity with strong OOD prototypes
        """

        # if the flag is 0, the prototype pool is empty, return None.
        if not self.flag:
            return None

        # compute the cosine similarity between the features and the strong OOD prototypes.
        out = torch.mm(x, self.memory.t())

        if all == True: # if all is True, return the cosine similarity with all the strong OOD prototypes.
            return out
        else: # if all is False, return the cosine similarity with the nearest strong OOD prototype.
            return torch.max(out, dim=1)[0].unsqueeze(1)

    def update_pool(self, feature):
        """
        Append and delete strong OOD prototypes
        """

        if not self.flag: # if the flag is 0, the prototype pool is empty, use the feature to init the prototype pool
            self.register_buffer("memory", feature.detach())
            self.flag = 1
        else:
            # if the number of strong OOD prototypes is less than the maximum count of strong OOD prototypes, append the feature to the prototype pool
            if self.memory.shape[0] < self.max_length:
                self.memory = torch.cat([self.memory, feature.detach()], dim=0)
            else: # else then delete the earlest appended strong OOD prototype and append the feature to the prototype pool
                self.memory = torch.cat([self.memory[1:], feature.detach()], dim=0)
        self.memory = F.normalize(self.memory)

class mask():
    def __init__(self, known_percentage_threshold, unknown_percentage_threshold, N_init):
        self.known_percentage_threshold = known_percentage_threshold
        self.unknown_percentage_threshold = unknown_percentage_threshold

        self.tau_low = None
        self.tau_low_list = []
        self.tau_high = None
        self.tau_high_list = []

        self.count = 0
        self.N_init = N_init

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_mask(self, likelihood):
        logger = logging.getLogger(__name__)
        entropy_values = calculate_entropy_from_likelihood(likelihood)
        if self.count < self.N_init:
            logger.info("initializing tau values")
            # sort values (from small to big)
            sorted_A, _ = torch.sort(entropy_values)
            threshold_idx_known = math.ceil(len(sorted_A) * (self.known_percentage_threshold))
            threshold_a = sorted_A[threshold_idx_known]
            self.tau_low_list.append(threshold_a)
            tau_low = torch.tensor(self.tau_low_list)
            threshold_idx_unknown = math.floor(len(sorted_A) * (self.unknown_percentage_threshold))
            threshold_b = sorted_A[threshold_idx_unknown]
            self.tau_high_list.append(threshold_b)
            tau_high = torch.tensor(self.tau_high_list)
            self.tau_low = torch.mean(tau_low)
            self.tau_high = torch.mean(tau_high)
            self.count = self.count + 1
        
        # NOTE: the two while loops fail if tau_low is 0 (and the next while when tau_high is 1), so we chose the smallest/largest possible values instead

        # determine the threshold value for the percentage_threshold values
        known_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
        known_mask[entropy_values < self.tau_low] = True
        tau_low = self.tau_low

        if torch.sum(known_mask).item() <= 1:
            tau_low = torch.min(entropy_values)
            known_mask = entropy_values <= tau_low

        # while torch.sum(known_mask).item() <= 1:
        #     logger.info("adjusting tau_low")
        #     tau_low += self.tau_low
        #     known_mask = entropy_values <= tau_low

        unknown_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
        unknown_mask[entropy_values > self.tau_high] = True
        tau_high = self.tau_high

        # while torch.sum(unknown_mask).item() <= 1:
        #     logger.info("adjusting tau_high")
        #     tau_high -= self.tau_high
        #     unknown_mask = entropy_values >= tau_high

        if torch.sum(unknown_mask).item() <= 1:
            tau_high = torch.max(entropy_values)
            unknown_mask = entropy_values >= tau_high 

        both_true = torch.logical_and(known_mask, unknown_mask)
        unknown_mask[both_true] = False

        rejection_mask = (known_mask | unknown_mask)

        return known_mask, unknown_mask, rejection_mask

class GaussianMixtureModel():
    def __init__(self, source_class_num):
        self.source_class_num = source_class_num
        self.batch_weight = torch.zeros(source_class_num, dtype=torch.float)
        self.mu = None
        self.C = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def soft_update(self, feat, posterior):
        # Set the desired data type
        dtype = torch.float64  # You can use torch.float128 if supported
        torch.set_printoptions(threshold=float('inf'))

        posterior = posterior.to(dtype)
        feat = feat.to(device=self.device, dtype=dtype)
        self.batch_weight = self.batch_weight.to(device=self.device, dtype=dtype)

        # ---------- Calculate mu ----------
        # Calculate the sum of the posteriors
        batch_weight_new = torch.zeros(posterior.shape[1], device=self.device, dtype=dtype)
        batch_weight_new = batch_weight_new + torch.sum(posterior, dim=0)
        batch_weight_new = batch_weight_new + self.batch_weight

        # Calculate the sum of the weighted features
        weighted_sum = torch.matmul(posterior.T, feat)

        if self.mu != None:
            weighted_sum = torch.multiply(self.batch_weight.unsqueeze(1), self.mu) + weighted_sum

        # Calculate mu
        mu_new = weighted_sum / batch_weight_new[:, None]

        # ---------- Calculate the Covariance Matrices ----------
        # Calculate the sum of the outer product
        differences = feat.unsqueeze(1) - mu_new.unsqueeze(0)

        outer_prods = torch.einsum('nmd,nme->nmde', differences, differences)
        epsilon = 1e-6
        eye = torch.eye(differences.shape[2], device=self.device).unsqueeze(0).unsqueeze(0)
        outer_prods = 0.5 * (outer_prods + outer_prods.transpose(-1, -2)) + epsilon * eye

        posterior_expanded = posterior.unsqueeze(-1).unsqueeze(-1)
        weighted_sum = torch.sum(posterior_expanded * outer_prods, dim=0)

        if self.C != None:
            weighted_sum = self.C * self.batch_weight.unsqueeze(1).unsqueeze(2) + weighted_sum

        # Calculate C
        C_new = weighted_sum / batch_weight_new[:, None, None]

        self.batch_weight = batch_weight_new
        self.mu = mu_new
        self.C = C_new

    def get_likelihood(self, feat, mu, C):
        logger = logging.getLogger(__name__)
        torch.set_printoptions(threshold=float('inf'))
        likelihood = torch.zeros((mu.shape[0], feat.shape[0]))

        # Compute the likelihood of the features for each class
        for i, (mean, cov) in enumerate(zip(mu, C)):
            mean = mean.cpu().detach().numpy() if isinstance(mean, torch.Tensor) else mean
            cov = cov.cpu().detach().numpy() if isinstance(cov, torch.Tensor) else cov
            feat = feat.cpu() if feat.is_cuda else feat
            rv = multivariate_normal(mean, cov, allow_singular=True)
            vals = rv.logpdf(feat)  # shape (N,) or scalar
            likelihood[i, :] = torch.from_numpy(np.array(vals)).type_as(likelihood)

        # for numerical stability
        maximum_likelihood = torch.max(likelihood).item()
        likelihood = likelihood - maximum_likelihood
        likelihood = torch.exp(likelihood)

        # normalize the likelihood
        likelihood = likelihood / torch.sum(likelihood, axis=0, keepdims=True)
        likelihood = likelihood.T

        return likelihood

    def get_labels(self, feat):
        likelihood = self.get_likelihood(feat, self.mu, self.C)
        max_values, max_indices = torch.max(likelihood, dim=1)

        return max_values, max_indices, likelihood

#### tta models ####

class Oracle(nn.Module):
    """
    Takes ground-truth labels and uses them for making predictions

    - O1: unknown samples are removed from predictions
    - O2: unknown samples are classified as unknown class and their entropy is maximized
    - O3: O1 + Kullback Leibler
    - O4: O2 + Kullback Leibler
    - O5: O1 + memory for more confident samples
    - O6: O2 + memory for more confident samples
    - O7: O3 + memory for more confident samples + memory for low confidence samples
    """

    def __init__(self, model, optimizer, adapted_model_key, steps=1, source_class_list=None, target_class_list=None, neg_mem_weight=None, kl_weight=None,
                 backbone_source=None, checkpoint_path=None, target_classnames_gsearch=None, cluster_params_path=None, source_classnames=None):

        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.adapted_model_key = adapted_model_key
        self.device = self.model.device
        self.source_class_list = source_class_list
        self.target_class_list = target_class_list
        self.neg_mem_weight = neg_mem_weight
        self.kl_weight = kl_weight
        self.num_classes = self.model.num_classes
        
        if self.kl_weight:
            logging.info(f"[DEBUG] setting up norm model for KL divergence with kl_weight: {self.kl_weight}")
            self.norm_model = load_unida_model(self.adapted_model_key, checkpoint_path, num_classes=self.num_classes, backbone_source=backbone_source,
                                               target_classnames_gsearch=target_classnames_gsearch, cluster_params_path=cluster_params_path,
                                               source_classnames=source_classnames).train()
            self.norm_model.requires_grad_(False)
            logging.info(f"[DEBUG] norm model loaded")
        
        self.steps = steps
        assert steps > 0, "oracle requires >= 1 step(s) to forward and update"

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

    def forward(self, x, gt_labels=None):
        if self.steps > 0:
            for _ in range(self.steps):
                if self.adapted_model_key == "tasc":
                    outputs, unknown_scores_dict = self.forward_and_adapt(x, self.adapted_model_key, self.model, self.optimizer, gt_labels=gt_labels)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs, unknown_scores_dict
                else:
                    outputs = self.forward_and_adapt(x, self.adapted_model_key, self.model, self.optimizer, gt_labels=gt_labels)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs
        else:
            self.model.eval()
            with torch.no_grad():
                if self.adapted_model_key == "tasc":
                    outputs, unknown_scores_dict = self.model(x)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs, unknown_scores_dict
                else:
                    outputs = self.model(x)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, adapted_model_key, model, optimizer, gt_labels=None):
        """
        Forward and adapt using cross-entropy against ground-truth labels
        assumes model outputs probabilities and optimizer is not sam
        """

        logging.info(f"[DEBUG] forward_and_adapt called with adapted_model_key: {adapted_model_key}")
        batch_size = int(x.size(0))
        logging.info(f"[DEBUG] batch_size: {batch_size}")
        logging.info(f"[DEBUG] gt_labels: {gt_labels.shape[0]}")

        if adapted_model_key == "tasc":
            outputs, unknown_scores_dict = model(x)  # probs [B, C]
            thr = self.thr_curr if self.thr_curr is not None else 0
            logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
        else:
            outputs = model(x)  # probs [B, C]

        known_mask = torch.isin(gt_labels, torch.tensor(self.source_class_list, device=gt_labels.device))
        unk_mask = ~known_mask
        n_known = int(known_mask.sum().item())
        logging.info(f"[DEBUG] adapting with {n_known} reliable samples out of {batch_size} samples")

        outputs_retraining = outputs[known_mask]
        labels_retraining  = gt_labels[known_mask]

        if self.neg_mem_weight:
            neg_outputs = outputs[unk_mask]
            logging.info(f"[DEBUG] using {neg_outputs.shape[0]} unknown samples for entropy maximization")
        
        n = outputs_retraining.shape[0] # retraining if at least 2 samples
        if n > 1:
            logging.info(f"[DEBUG] adapted on {n} samples with cross-entropy")
            labels_retraining = labels_retraining.to(dtype=torch.long, device=outputs_retraining.device)
            logp = torch.log(outputs_retraining.clamp_min(1e-12))
            loss = F.nll_loss(logp, labels_retraining)

            if self.neg_mem_weight:
                logging.info(f"[DEBUG] using {self.neg_mem_weight} as weight for entropy maximization")
                neg_entropy = entropy(neg_outputs)
                neg_loss = -neg_entropy.mean()
                loss += self.neg_mem_weight * neg_loss

            if self.kl_weight:
                logging.info(f"[DEBUG] using {self.kl_weight} as weight for KL divergence")
                output_norm, unknown_scores_dict_norm = self.norm_model(x)
                output_norm = output_norm[known_mask]
                logp_norm = torch.log(output_norm.clamp_min(1e-12))
                kl_loss = F.kl_div(logp, logp_norm)
                loss += self.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.info(f"[DEBUG] adapted on {n} samples with cross-entropy")
        else:
            logging.info(f"[DEBUG] no adaptation performed due to too few samples after masking: {n}")

        if adapted_model_key == "tasc":
            return outputs, unknown_scores_dict
        else:
            return outputs

    @staticmethod
    def configure_model(model, cfg_params_oracle):
        model.train()
        model.requires_grad_(False)

        logger = logging.getLogger(__name__)
        logging.info("[DEBUG] ---start: configure_model oracle---")

        # configure norm for eata updates: enable grad + force batch statisics
        logging.info(f"cfg_params_oracle: {cfg_params_oracle}")
        for m in model.modules():
            # logging.info(f"current module: {m}")
            if cfg_params_oracle in ["norm_only", "norm_and_lora", "all"]:
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

                    # SAFETY CHECK:
                    logging.info(f"[DEBUG] enabling grad for {m} (BatchNorm2d) and forcing batch statistics")
                if isinstance(m, nn.LayerNorm):
                    m.requires_grad_(True)

                    # SAFETY CHECK:
                    logging.info(f"[DEBUG] enabling grad for {m} (LayerNorm) and forcing batch statistics")
                    
            if cfg_params_oracle in ["norm_and_lora", "all"]:
                # if not isinstance(m, nn.BatchNorm2d) and not isinstance(m, nn.LayerNorm):
                if isinstance(m, LORAMultiheadAttention):
                    logging.info(f"current module: {m}")
                    m.requires_grad_(True)
            
            if cfg_params_oracle == "all":
                if not isinstance(m, nn.BatchNorm2d) and not isinstance(m, LORAMultiheadAttention):
                    logging.info(f"current module: {m}")
                    m.requires_grad_(True)
        
        logging.info("[DEBUG] ---end: configure_model oracle---")
        return model

    @staticmethod
    def collect_params(model, cfg_params_oracle):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        NOTE: other choices of parameterization are possible!
        """

        params = []
        names = []

        for nm, m in model.named_modules():
            if cfg_params_oracle in ["norm_only", "norm_and_lora", "all"]:
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in ["weight", "bias"]:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
                    logging.info(f"[DEBUG] collected params from {m}")
            
            if cfg_params_oracle in ["norm_and_lora", "all"]:
                if isinstance(m, LORAMultiheadAttention):
                    for np, p in m.named_parameters():
                        params.append(p)
                        names.append(f"{nm}.{np}")
                    logging.info(f"[DEBUG] collected params from {m}")

            if cfg_params_oracle == "all":
                if not isinstance(m, nn.BatchNorm2d) and not isinstance(m, LORAMultiheadAttention):
                    for np, p in m.named_parameters():
                        params.append(p)
                        names.append(f"{nm}.{np}")
                    logging.info(f"[DEBUG] collected params from {m}")

        return params, names

    @staticmethod
    def check_model(model):
        """
        Check model for compatability with oracle
        """

        is_training = model.training
        assert is_training, "oracle needs train mode: call model.train()"

        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)

        assert has_any_params, "oracle needs params to update: " \
                               "check which require grad"

        assert not has_all_params, "oracle should not update all params: " \
                                   "check which require grad"

        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "oracle needs normalization for its optimization"
    
class TENT(nn.Module):
    """
    TENT adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, adapted_model_key, steps=1, use_unk_classifier=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.adapted_model_key = adapted_model_key
        self.device = self.model.device
        self.use_unk_classifier = use_unk_classifier
        
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"

        self.num_classes = self.model.num_classes

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

    def forward(self, x):
        if self.steps > 0:
            for _ in range(self.steps):
                if self.adapted_model_key == "tasc":
                    outputs, unknown_scores_dict = self.forward_and_adapt(x, self.adapted_model_key, self.model, self.optimizer)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs, unknown_scores_dict
                else:
                    outputs = self.forward_and_adapt(x, self.adapted_model_key, self.model, self.optimizer)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs
        else:
            self.model.eval()
            with torch.no_grad():
                if self.adapted_model_key == "tasc":
                    outputs, unknown_scores_dict = self.model(x)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs, unknown_scores_dict
                else:
                    outputs = self.model(x)
                    logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
                    return outputs

    @torch.enable_grad()  
    def forward_and_adapt(self, x, adapted_model_key, model, optimizer):
        """
        Forward and adapt model on batch of data. Measure entropy of the model 
        prediction, take gradients, and update params.
        """

        batch_size = int(x.size(0))
        logging.info(f"[DEBUG] batch_size: {batch_size}")

        if adapted_model_key == "tasc":
            logging.info(f"[DEBUG] batch size: {batch_size}")
            outputs, unknown_scores_dict = model(x)

            # filtering out unreliable samples according to model
            if self.use_unk_classifier:
                known_scores = -unknown_scores_dict["UniMS"]
                known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                thr = self.thr_curr if self.thr_curr is not None else 0
                logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                known_mask = known_scores >= thr       
                logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples out of {batch_size} samples")
                outputs_retraining = outputs[known_mask]
            else:
                outputs_retraining = outputs

            if outputs_retraining.shape[0] > 1:
                logging.info(f"[DEBUG] adapting with {outputs_retraining.shape[0]} samples")
                loss = entropy(outputs_retraining).mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                logging.info(f"[DEBUG] no adaptation performed, num samples for adaptation: {outputs_retraining.shape[0]} samples")

            return outputs, unknown_scores_dict
            
        else:
            if adapted_model_key != "tasc":
                if batch_size > 1: # batch normalization works with batches > 1
                    logging.info(f"[DEBUG] adapting with batch size: {batch_size}")
                    outputs = model(x)

                    if self.use_unk_classifier:
                        entropys = entropy(outputs)
                        known_mask = entropys < 0.55 # threshold used in LEAD and GLC
                        outputs_retraining = outputs[known_mask]
                    else:
                        outputs_retraining = outputs
                    
                    if outputs_retraining.shape[0] > 1:
                        logging.info(f"[DEBUG] adapting with {outputs_retraining.shape[0]} samples")
                        loss = entropy(outputs).mean(0)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        logging.info(f"[DEBUG] no adaptation performed, num samples for adaptation: {outputs_retraining.shape[0]} samples")

                else:
                    logging.info(f"[DEBUG] no adaptation performed, batch size: {batch_size}")                    
                    model.eval()
                    with torch.no_grad():
                        outputs = model(x)
                
                return outputs
            else:
                raise ValueError(f"{adapted_model_key} should not be here!")

    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)

        logger = logging.getLogger(__name__)
        logging.info("[DEBUG] ---start: configure_model tent---")

        # configure norm for eata updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

                # SAFETY CHECK:
                logging.info(f"[DEBUG] enabling grad for {m} (BatchNorm2d) and forcing batch statistics")
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

                # SAFETY CHECK:
                logging.info(f"[DEBUG] enabling grad for {m} (LayerNorm) and forcing batch statistics")
        
        logging.info("[DEBUG] ---end: configure_model tent---")
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        NOTE: other choices of parameterization are possible!
        """

        params = []
        names = []

        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        return params, names

    @staticmethod
    def check_model(model):
        """
        Check model for compatability with tent
        """

        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"

        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)

        assert has_any_params, "tent needs params to update: " \
                               "check which require grad"

        assert not has_all_params, "tent should not update all params: " \
                                   "check which require grad"

        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"

class EATA(nn.Module):
    """
    EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, adapted_model_key, fishers=None, fisher_alpha=2000.0, steps=1,
                 e_margin=math.log(1000) / 2 - 1, d_margin=0.05, use_unk_classifier=None):
        super().__init__()
        
        self.model = model
        self.model = self.configure_model(model)
        self.optimizer = optimizer
        self.adapted_model_key = adapted_model_key
        self.steps = steps
        self.num_classes = self.model.num_classes
        self.device = self.model.device
        self.use_unk_classifier = use_unk_classifier
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)

        self.os_queue = []

        self.fishers = fishers  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha  # trade-off \beta for two losses (Eqn. 8)

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

    def forward(self, x):        
        batch_size = int(x.size(0))
        logging.info(f"[DEBUG] batch_size: {batch_size}")

        if self.steps > 0:
            for _ in range(self.steps):
                if self.adapted_model_key == "tasc":                
                    logging.info(f"[DEBUG] adapting with batch size: {batch_size}")
                    outputs, unknown_model_dict, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x, self.model,
                                                                                                self.optimizer,
                                                                                                self.fishers, self.e_margin,
                                                                                                self.current_model_probs,
                                                                                                fisher_alpha=self.fisher_alpha,
                                                                                                num_samples_update=self.num_samples_update_2,
                                                                                                d_margin=self.d_margin)
                else:
                    if batch_size > 1:                    
                        logging.info(f"[DEBUG] adapting with batch size: {batch_size}")
                        outputs, num_counts_2, num_counts_1, updated_probs = self.forward_and_adapt(x, self.model,
                                                                                                self.optimizer,
                                                                                                self.fishers, self.e_margin,
                                                                                                self.current_model_probs,
                                                                                                fisher_alpha=self.fisher_alpha,
                                                                                                num_samples_update=self.num_samples_update_2,
                                                                                                d_margin=self.d_margin)
                    else:
                        logging.info(f"[DEBUG] no adaptation performed, batch size: {batch_size}")
                        self.model.eval()
                        with torch.no_grad():
                            outputs = self.model(x)
                            updated_probs = self.update_model_probs(self.current_model_probs, outputs)
                        
                        num_counts_2, num_counts_1 = 0, 0

                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                if self.adapted_model_key == "tasc":
                    outputs, unknown_model_dict = self.model(x)
                else:
                    outputs = self.model(x)

        # returning model outputs
        logging.info(f"[DEBUG]: final output shape: {outputs.shape[0]}")
        if self.adapted_model_key == "tasc":
            return outputs, unknown_model_dict
        else:
            return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0,
                            d_margin=0.05, scale_factor=2, num_samples_update=0):
        """
        Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: 
        1. model outputs; 
        2. the number of reliable and non-redundant samples; 
        3. the number of reliable samples;
        4. the moving average probability vector over all previous samples
        """

        with torch.no_grad(): 
            # forward
            if self.adapted_model_key == "tasc":
                outputs, unknown_model_dict = model(x)

                if self.use_unk_classifier: # filtering out unreliable samples according to model
                    known_scores = -unknown_model_dict["UniMS"]
                    known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                    thr = self.thr_curr if self.thr_curr is not None else 0
                    logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                    known_mask = known_scores >= thr       
                    logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                else:
                    known_mask = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)

            else:
                outputs = model(x)
                
                if self.use_unk_classifier:
                    entropys = entropy(outputs)
                    known_mask = entropys < 0.55 # threshold used in LEAD and GLC
                    outputs_retraining = outputs[known_mask]
                else:
                    known_mask = torch.ones(x.shape[0], dtype=torch.bool, device=self.device)
            
            outputs_retraining = outputs[known_mask]
            
            logging.info(f"[DEBUG] adapting with {outputs_retraining.shape[0]} samples")
            entropys = entropy(outputs_retraining)
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys < e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0] > -0.1)
            entropys = entropys[filter_ids_1]

            # filter redundant samples
            if current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), # here is where current_model_probs are used
                                                        outputs_retraining[filter_ids_1].softmax(1), dim=1)
                logging.info(f"[DEBUG] cosine_similarities: {cosine_similarities}")
                logging.info(f"[DEBUG] d_margin: {d_margin}")
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = self.update_model_probs(current_model_probs, outputs_retraining[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = self.update_model_probs(current_model_probs, outputs_retraining[filter_ids_1].softmax(1))
        
        logging.info(f"[DEBUG] EATA ids2: {ids2}, ids1: {ids1}")
        if ids2[0].size(0) > 0:
            x_sel = x[known_mask].index_select(0, ids2[0])
            if self.adapted_model_key == "tasc":
                outputs_sel, _ = model(x_sel)   # now build a graph but for the small subset
            else:
                outputs_sel = model(x_sel)

            ent_sel = entropy(outputs_sel)                    # has graph
            coeff = 1.0 / torch.exp(ent_sel.detach() - e_margin)   # weights w/o grads
            ent_sel = ent_sel.mul(coeff)  # reweight entropy losses for diff. samples
            loss = ent_sel.mean(0)

            if fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in fishers:
                        ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1]) ** 2).sum()
                loss += ewc_loss

            if x[known_mask][ids1[0]][ids2[0]].size(0) >= 2:
                loss.backward()
                optimizer.step()

        optimizer.zero_grad()

        # coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        # entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        # loss = entropys.mean(0)
        
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """

        if self.adapted_model_key == "tasc":
            return outputs, unknown_model_dict, entropys.size(0), filter_ids_1[0].size(0), updated_probs
        else:
            return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs

    def update_model_probs(self, current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    @staticmethod
    def configure_model(model):
        """
        Configure model for use with eata
        """

        logging.info("[DEBUG] ---start: configure_model eata---")
        
        # train mode, because eata optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what eata updates
        model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        
        logging.info("[DEBUG] ---end: configure_model eata---")
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

class SoTTA(nn.Module):
    """
    Screeening-out Test-Time Adaptation
    """

    def __init__(self, model, optimizer, adapted_model_key, ConfThreshold, use_unk_classifier=None):
        super(SoTTA, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.adapted_model_key = adapted_model_key
        assert (isinstance(self.optimizer, SAM))
        self.memory_size = 64
        self.ConfThreshold = ConfThreshold
        self.num_classes = self.model.num_classes
        self.memory = HUS(self.memory_size, self.num_classes, ConfThreshold)
        self.use_unk_classifier = use_unk_classifier

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

        self.device = self.model.device

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        NOTE: other choices of parameterization are possible!
        """

        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if "layer4" in nm:
                continue
            if "conv5_x" in nm:
                continue
            if "blocks.9" in nm:
                continue
            if "blocks.10" in nm:
                continue
            if "blocks.11" in nm:
                continue
            if "norm." in nm:
                continue
            if nm in ["norm"]:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    @staticmethod
    def configure_model(model, momentum=0.2):
        logging.info("[DEBUG] ---start: configure_model sotta---")
        for param in model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.momentum = momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        logging.info("[DEBUG] ---end: configure_model sotta---")
        return model

    def forward(self, x):
        self.model.eval()

        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(x)
            
            if self.use_unk_classifier: # filtering out unreliable samples according to model
                known_scores = -unknown_scores_dict["UniMS"]
                known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                thr = self.thr_curr if self.thr_curr is not None else 0
                logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                known_mask = known_scores >= thr       
                logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                probs_retraining = probs[known_mask]
            else:
                probs_retraining = probs

        else:
            probs = self.model(x)

            if self.use_unk_classifier:
                entropies = entropy(probs)
                known_mask = entropies < 0.55 # threshold used in LEAD and GLC
                probs_retraining = probs[known_mask]
            else:
                probs_retraining = probs
        
        logging.info(f"[DEBUG] adapting with {probs_retraining.shape[0]} samples")
        
        # add confident samples to memory
        confidences, pseudo_labels = torch.max(probs_retraining, dim=1)
        if self.use_unk_classifier:
            logging.info(f"[DEBUG] x size after filtering: {x[known_mask].shape[0]} samples")
            for i in range(x[known_mask].shape[0]):
                self.memory.add_instance([x[known_mask][i], pseudo_labels[i], confidences[i]])
        else:
            logging.info(f"[DEBUG] x size: {x.shape[0]} samples")
            for i in range(x.shape[0]):
                self.memory.add_instance([x[i], pseudo_labels[i], confidences[i]])

        # get features from memory and adapt
        feats = self.memory.get_memory()
        try:
            # if feats.shape[0] == 0 or feats is None:
            if feats is None:
                if self.adapted_model_key == "tasc":
                    return probs, unknown_scores_dict
                else:
                    return probs
            elif feats.shape[0] == 1:
                self.model.eval()
            else:
                self.model.train()
        except:
            logging.warning("feats.shape[0] == 0")
            if self.adapted_model_key == "tasc":
                return probs, unknown_scores_dict
            else:
                return probs
        
        # adaptation step
        batch_size = 0 if feats is None else (feats.shape[0] if hasattr(feats, "shape") else len(feats))
        logging.info(f"[DEBUG] SoTTA batch_size: {batch_size}")
        if batch_size >= 2:
            self.optimize(feats)

        logging.info(f"[DEBUG]: final output shape: {probs.shape[0]}")
        if self.adapted_model_key == "tasc":
            return probs, unknown_scores_dict
        else:
            return probs

    @torch.enable_grad()
    def optimize(self, data):

        self.model.train()
        self.optimizer.zero_grad()

        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
        else:
            probs = self.model(data)
        
        loss = entropy(probs).mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        self.model.train()
        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
        else:
            probs = self.model(data)

        loss = entropy(probs).mean()
        loss.backward()
        self.optimizer.second_step(zero_grad=True)   

class SoTTA_v2(nn.Module):
    """
    Screeening-out Test-Time Adaptation
    """

    def __init__(self, model, optimizer, adapted_model_key, ConfThreshold, use_unk_classifier=None, neg_mem_weight=None):
        super(SoTTA_v2, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.adapted_model_key = adapted_model_key
        assert (isinstance(self.optimizer, SAM))
        self.memory_size = 64
        self.ConfThreshold = ConfThreshold
        self.num_classes = self.model.num_classes
        self.memory = HUS(self.memory_size, self.num_classes, ConfThreshold)
        self.neg_memory = HUS(self.memory_size, self.num_classes, None) # memory of unk class samples 
        self.neg_mem_weight = neg_mem_weight # weight for negative memory samples in adaptation
        self.use_unk_classifier = use_unk_classifier

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

        self.device = self.model.device

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        NOTE: other choices of parameterization are possible!
        """

        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if "layer4" in nm:
                continue
            if "conv5_x" in nm:
                continue
            if "blocks.9" in nm:
                continue
            if "blocks.10" in nm:
                continue
            if "blocks.11" in nm:
                continue
            if "norm." in nm:
                continue
            if nm in ["norm"]:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names

    @staticmethod
    def configure_model(model, momentum=0.2):
        logging.info("[DEBUG] ---start: configure_model sotta---")
        for param in model.parameters():  # initially turn off requires_grad for all
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
                module.momentum = momentum

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.LayerNorm):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        logging.info("[DEBUG] ---end: configure_model sotta---")
        return model

    def forward(self, x):
        self.model.eval()

        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(x)
            
            if self.use_unk_classifier: # filtering out unreliable samples according to model
                known_scores = -unknown_scores_dict["UniMS"]
                known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                thr = self.thr_curr if self.thr_curr is not None else 0
                logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                known_mask = known_scores >= thr       
                logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                probs_retraining = probs[known_mask]

                # adding samples predicted as unknown to the negative memory # TODO: also add samples with low confidence to the negative memory
                unk_mask = known_scores < thr
                
                logging.info(f"[DEBUG] adding {torch.sum(unk_mask).item()} samples predicted as unknown to the negative memory")
            else:
                probs_retraining = probs

        else:
            probs = self.model(x)

            if self.use_unk_classifier:
                entropies = entropy(probs)
                known_mask = entropies < 0.55 # threshold used in LEAD and GLC
                probs_retraining = probs[known_mask]
            else:
                probs_retraining = probs
        
        logging.info(f"[DEBUG] adapting with {probs_retraining.shape[0]} samples")
        
        # add confident samples to memory
        confidences, pseudo_labels = torch.max(probs_retraining, dim=1)

        # all confidences
        all_confidences, all_pseudo_labels = torch.max(probs, dim=1)

        # # finding low confidence samples
        # low_conf_mask = all_confidences < self.ConfThreshold
        # logging.info(f"[DEBUG] adding {torch.sum(low_conf_mask).item()} low confidence samples to the negative memory") # could also be low entropy

        # unk_mask = torch.logical_or(unk_mask, low_conf_mask)
        # logging.info(f"[DEBUG] final num samples for neg memory {torch.sum(unk_mask).item()} low confidence samples to the negative memory")

        if self.use_unk_classifier:
            logging.info(f"[DEBUG] x size after filtering: {x[known_mask].shape[0]} samples")
            for i in range(x[known_mask].shape[0]):
                self.memory.add_instance([x[known_mask][i], pseudo_labels[i], confidences[i]])
            for j in range(x[unk_mask].shape[0]):
                self.neg_memory.add_instance([x[unk_mask][j], all_pseudo_labels[j], all_confidences[j]])

            # get features from negative memory and adapt
            neg_feats = self.neg_memory.get_memory()
        else:
            logging.info(f"[DEBUG] x size: {x.shape[0]} samples")
            for i in range(x.shape[0]):
                self.memory.add_instance([x[i], pseudo_labels[i], confidences[i]])

        # get features from memory and adapt
        feats = self.memory.get_memory()

        try:
            # if feats.shape[0] == 0 or feats is None:
            if feats is None:
                if self.adapted_model_key == "tasc":
                    return probs, unknown_scores_dict
                else:
                    return probs
            elif feats.shape[0] == 1:
                self.model.eval()
            else:
                self.model.train()
        except:
            logging.warning("feats.shape[0] == 0")
            if self.adapted_model_key == "tasc":
                return probs, unknown_scores_dict
            else:
                return probs
        
        # adaptation step
        batch_size = 0 if feats is None else (feats.shape[0] if hasattr(feats, "shape") else len(feats))
        logging.info(f"[DEBUG] SoTTA batch_size: {batch_size}")
        if batch_size >= 2:
            if self.use_unk_classifier:
                if neg_feats is None or neg_feats.shape[0] < 2:
                    self.optimize(feats)
                else:
                    self.optimize_w_neg(feats, neg_feats)
            else:
                self.optimize(feats)

        logging.info(f"[DEBUG]: final output shape: {probs.shape[0]}")
        if self.adapted_model_key == "tasc":
            return probs, unknown_scores_dict
        else:
            return probs

    @torch.enable_grad()
    def optimize(self, data):

        self.model.train()
        self.optimizer.zero_grad()

        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
        else:
            probs = self.model(data)
        
        loss = entropy(probs).mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        self.model.train()
        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
        else:
            probs = self.model(data)

        loss = entropy(probs).mean()
        loss.backward()
        self.optimizer.second_step(zero_grad=True)    

    @torch.enable_grad()
    def optimize_w_neg(self, data, data_neg):

        logging.info(f"[DEBUG] optimize_neg: data size: {data.shape[0]}, data_neg size: {data_neg.shape[0]}")

        self.model.train()
        self.optimizer.zero_grad()

        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
            neg_probs, neg_unknown_scores_dict = self.model(data_neg)
        else:
            probs = self.model(data)
            neg_probs = self.model(data_neg)
        
        loss_pos = entropy(probs).mean()
        loss_neg = -entropy(neg_probs).mean() # negative entropy loss for negative samples
        loss = loss_pos + self.neg_mem_weight * loss_neg # balancing the two losses
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        self.model.train()
        if self.adapted_model_key == "tasc":
            probs, unknown_scores_dict = self.model(data)
            neg_probs, neg_unknown_scores_dict = self.model(data_neg)
        else:
            probs = self.model(data)
            neg_probs = self.model(data_neg)

        loss_pos = entropy(probs).mean()
        loss_neg = -entropy(neg_probs).mean() # negative entropy loss for negative samples
        loss = loss_pos + self.neg_mem_weight * loss_neg # balancing the two losses
        loss.backward()
        self.optimizer.second_step(zero_grad=True)    

        logging.info(f"[DEBUG] optimize_neg: first step done, start negative step")

class OWTTT(nn.Module):
    def __init__(self, model, model_key, optimizer, num_classes=10, ce_scale=0.2, da_scale=1, delta=0.1, queue_length=512,
                 max_prototypes=100, source_memory=None, source_distribution=None):

        # TODO: figure out possible memory leakages
        super(OWTTT, self).__init__()

        self.model = model
        self.adapted_model_key = model_key
        self.optimizer = optimizer

        # NOTE: not needed as we reinitialize the class per model adaptation
        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

        self.device = self.model.device
        self.ood_memory = Prototype_Pool(max=max_prototypes)
        self.queue_training = []
        self.queue_inference = []
        self.ema_total_n = 0.
        self.ema_distribution = {}
        self.ema_distribution["mu"] = torch.zeros(self.model.output_dim).float()
        self.ema_distribution["cov"] = torch.zeros(self.model.output_dim, self.model.output_dim).float()

        self.ce_scale = ce_scale
        self.da_scale = da_scale
        self.delta = delta
        self.queue_length = queue_length
        self.max_prototypes = max_prototypes
        self.source_memory = source_memory  # source prototypes
        bias = source_distribution["cov"].max().item() / 30.
        self.template_cov = torch.eye(self.model.output_dim).cuda() * bias
        self.source_distribution = source_distribution  # a dict containing the mu and cov matrix of the source domain

        self.num_classes = num_classes
        if num_classes == 10:
            self.loss_scale = 0.05
        else:
            self.loss_scale = 0.05

        self.threshold_range = np.arange(0, 1, 0.01)  # use for consequent operation

    @torch.enable_grad()
    def forward(self, inputs):
        batch_size = int(inputs.size(0))
        logging.info(f"[DEBUG] batch_size: {batch_size}")
        if self.adapted_model_key == "tasc":
            logging.info(f"[DEBUG] adapting with batch size: {batch_size}")
            return self.adapt_and_forward(inputs)
        else:
            if batch_size > 1:
                logging.info(f"[DEBUG] adapting with batch size: {batch_size}")
                return self.adapt_and_forward(inputs)
            else:
                logging.info(f"[DEBUG] no adaptation performed, batch size: {batch_size}")
                self.model.eval()
                with torch.no_grad():
                    softmax_logit = self.model(inputs)
                    return softmax_logit

    def adapt_and_forward(self, inputs):
        logger = logging.getLogger(__name__)

        self.model.eval()
        feat, _ = self.model(inputs, return_feats=True)
        feat_norm = F.normalize(feat)
        self.optimizer.zero_grad()

        cos_sim_src = self.source_memory(feat_norm, all=True)
        cos_sim_ood = self.ood_memory(feat_norm)

        if cos_sim_ood is not None:
            cos_sim = torch.cat([cos_sim_src, cos_sim_ood], dim=1)
        else:
            cos_sim = cos_sim_src

        logits = cos_sim / self.delta

        with torch.no_grad():
            ood_score, pseudo_labels = 1 - cos_sim.max(dim=-1)[0], cos_sim.max(dim=-1)[1]
            ood_score_src = 1 - cos_sim_src.max(dim=-1)[0]

            self.queue_training.extend(ood_score_src.detach().cpu().tolist())
            self.queue_training = self.queue_training[-self.queue_length:]

            criterias = [compute_os_variance(np.array(self.queue_training), th) for th in self.threshold_range]
            best_threshold_ood = self.threshold_range[np.argmin(criterias)]
            seen_mask = (ood_score_src < best_threshold_ood)
            unseen_mask = (ood_score_src >= best_threshold_ood)

            if unseen_mask.sum().item() != 0:
                criterias = [compute_os_variance(ood_score[unseen_mask].detach().cpu().numpy(), th) for th in
                            self.threshold_range]
                best_threshold_exp = self.threshold_range[np.argmin(criterias)]

                # append new strong OOD prototypes to the prototype pool.
                append_prototypes(self.ood_memory, feat, cos_sim_src, best_threshold_ood, best_threshold_exp)

            len_memory = len(cos_sim[0])
            if len_memory != self.num_classes:
                if seen_mask.sum().item() != 0:
                    pseudo_labels[seen_mask] = cos_sim_src[seen_mask].max(dim=-1)[1]
                if unseen_mask.sum().item() != 0:
                    pseudo_labels[unseen_mask] = self.num_classes
            else:
                pseudo_labels = cos_sim_src[seen_mask].max(dim=-1)[1]

        loss = torch.tensor(0.).cuda()
        # ------distribution alignment------
        if seen_mask.sum().item() != 0:
            self.model.train()
            feat_global = self.model(inputs, return_feats=True)[0]
            # Global Gaussian
            b = feat_global.shape[0]
            self.ema_total_n += b
            alpha = 1. / 1280 if self.ema_total_n > 1280 else 1. / self.ema_total_n
            delta_pre = (feat_global - self.ema_distribution["mu"].cuda())
            delta = alpha * delta_pre.sum(dim=0)
            tmp_mu = self.ema_distribution["mu"].cuda() + delta
            tmp_cov = self.ema_distribution["cov"].cuda() + alpha * (
                    delta_pre.t() @ delta_pre - b * self.ema_distribution["cov"].cuda()) - delta[:, None] @ delta[None,
                                                                                                            :]
            self.ema_distribution["mu"] = tmp_mu.detach().cpu()
            self.ema_distribution["cov"] = tmp_cov.detach().cpu()

            source_domain = torch.distributions.MultivariateNormal(self.source_distribution["mu"],
                                                                   self.source_distribution["cov"] + self.template_cov)
            
            target_domain = torch.distributions.MultivariateNormal(tmp_mu, tmp_cov + self.template_cov)
            loss += self.da_scale * (
                    torch.distributions.kl_divergence(source_domain, target_domain) + torch.distributions.kl_divergence(
                target_domain, source_domain)) * self.loss_scale

        if len_memory != self.num_classes and seen_mask.sum().item() != 0 and unseen_mask.sum().item() != 0:
            a, idx1 = torch.sort((ood_score_src[seen_mask]), descending=True)
            filter_down = a[-int(seen_mask.sum().item() * (1 / 2))]
            a, idx1 = torch.sort((ood_score_src[unseen_mask]), descending=True)
            filter_up = a[int(unseen_mask.sum().item() * (1 / 2))]
            for j in range(len(pseudo_labels)):
                if ood_score_src[j] >= filter_down and seen_mask[j]:
                    seen_mask[j] = False
                if ood_score_src[j] <= filter_up and unseen_mask[j]:
                    unseen_mask[j] = False

        if len_memory != self.num_classes:
            entropy_seen = nn.CrossEntropyLoss()(logits[seen_mask, :self.num_classes], pseudo_labels[seen_mask])
            entropy_unseen = nn.CrossEntropyLoss()(logits[unseen_mask], pseudo_labels[unseen_mask])
            loss += self.ce_scale * (entropy_seen + entropy_unseen) / 2 

        try:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        except:
            logger.info("can not backward")
        torch.cuda.empty_cache()

        ####-------------------------- Test ----------------------------####
        # TODO: update output for tasc and other models based on calculate_metrics function
        with torch.no_grad():
            self.model.eval()
            feats = self.model(inputs, return_feats=True)[0]
            cos_sim_src = self.source_memory(F.normalize(feats), all=True)
            softmax_logit = (cos_sim_src / self.delta).softmax(dim=-1)

        if self.adapted_model_key == "tasc":
            with torch.no_grad():
                _, unknown_scores_dict = self.model(inputs, return_feats=False)
            return softmax_logit, unknown_scores_dict
        else:
            return softmax_logit

class STAMP(nn.Module):
    """
    Implemented from commit https://github.com/yuyongcan/STAMP/tree/55b116c993417f9672fb43840ec844f0eb337c46
    """
    
    def __init__(self, model, model_key, checkpoint_path, backbone_source, optimizer, alpha,
                 target_classnames_gsearch=None, cluster_params_path=None, source_classnames=None, 
                 device="cuda", consistency_filtering=True, use_unk_classifier=None):
        super(STAMP, self).__init__()
        
        # loading model and optimizer
        self.device = device
        self.adapted_model_key = model_key
        self.optimizer = optimizer
        self.load_base_model(checkpoint_path, model, backbone_source, target_classnames_gsearch, 
                             cluster_params_path, source_classnames)
        self.consistency_filtering = consistency_filtering
        self.use_unk_classifier = use_unk_classifier

        # setting up TTA parameters
        self.alpha = alpha
        self.margin = alpha * math.log(self.num_classes)

        logging.info(f"[DEBUG] STAMP model configured with num_class = {self.num_classes}, alpha = {self.alpha}, margin = {self.margin}") # SAFETY CHECK:

        self.mem = RBM(64, self.num_classes)
        if self.num_classes == 1000:
            self.max_iter = 750
        else:
            self.max_iter = 150
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

    def load_base_model(self, checkpoint_path, model, backbone_source, target_classnames_gsearch, 
                        cluster_params_path, source_classnames):
        """
        Obtains model and optimizer states for reset
        """

        self.model = self.configure_model(model)
        self.num_classes = self.model.num_classes

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr
            self.norm_model = load_unida_model(self.adapted_model_key, checkpoint_path, num_classes=self.num_classes, backbone_source=backbone_source,
                                               target_classnames_gsearch=target_classnames_gsearch, cluster_params_path=cluster_params_path,
                                               source_classnames=source_classnames).train()
        else:
            self.norm_model = load_unida_model(self.adapted_model_key, checkpoint_path, num_classes=self.num_classes, backbone_source=backbone_source).train()
        self.norm_model.requires_grad_(False)

    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)

        logger = logging.getLogger(__name__)
        logger.info("[DEBUG] ---start: configure_model---")

        # configure norm for eata updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

                # SAFETY CHECK:
                logger.info(f"[DEBUG] enabling grad for {m} (BatchNorm2d) and forcing batch statistics")
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

                # SAFETY CHECK:
                logger.info(f"[DEBUG] enabling grad for {m} (LayerNorm) and forcing batch statistics")
        
        logger.info("[DEBUG] ---end: configure_model---")
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []
        logger = logging.getLogger(__name__)

        logger.info("[DEBUG] ---start: collect_params---")
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if "layer4" in nm:
                continue
            if "conv5_x" in nm:
                continue
            if "blocks.9" in nm:
                continue
            if "blocks.10" in nm:
                continue
            if "blocks.11" in nm:
                continue
            if "norm." in nm:
                continue
            if nm in ["norm"]:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        # SAFETY CHECK:
        # logger.info(f"[DEBUG] collected {len(params)} parameters for STAMP adaptation: {names}")
        logger.info("[DEBUG] ---end: collect_params---")

        return params, names

    def forward(self, input_imgs):
        logger = logging.getLogger(__name__)

        if self.adapted_model_key == "tasc":
            output, unknown_scores_dict = self.update_memory(input_imgs) 
            if len(self.mem) != 0:
                self.adapt()
            return output, unknown_scores_dict
        else:
            batch_size = int(input_imgs[0].size(0))
            logging.info(f"[DEBUG] batch_size: {batch_size}")
            if batch_size > 1:
                output = self.update_memory(input_imgs) 
                if len(self.mem) != 0:
                    self.adapt()
                return output
            else:
                logging.info(f"[DEBUG] no adaptation performed, batch size: {batch_size}")
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input_imgs[0]) # input_imgs[0] is the original image
                return output

    def update_memory(self, input_imgs):
        """
        Selects low-entropy samples from the input images and appends them to the memory
        """

        input_imgs_origin = input_imgs[0] # x is a list of augmented images, first is the original image

        logger = logging.getLogger(__name__)
        logger.info("[DEBUG] ---start: update_memory---")
        outputs = []
        self.model.train() # NOTE: this seems to be implemented for safety, but not sure if needed

        with torch.no_grad():
            if self.adapted_model_key == "tasc":
                output_origin, unknown_scores_dict = self.model(input_imgs_origin) # predictions after update

                if self.use_unk_classifier: # filtering out unreliable samples according to model
                    known_scores = -unknown_scores_dict["UniMS"]
                    known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                    thr = self.thr_curr if self.thr_curr is not None else 0
                    logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                    known_mask = known_scores >= thr       
                    output_origin_retraining = output_origin[known_mask]
                    logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                else:
                    output_origin_retraining = output_origin
            else:
                output_origin = self.model(input_imgs_origin) # predictions after update

                if self.use_unk_classifier:
                    entropies = entropy(output_origin)
                    known_mask = entropies < 0.55 # threshold used in LEAD and GLC
                    output_origin_retraining = output_origin[known_mask]
                    logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                else:
                    output_origin_retraining = output_origin
                            
        # SAFETY CHECK:
        # logging.info(f"[DEBUG] origin probs: {output_origin}")
        # logging.info(f"[DEBUG] origin entropys: {entropy(output_origin)}")
        
        # SAFETY CHECK:
        normalized_entropies = entropy(output_origin_retraining)/np.log(self.num_classes) # [N]
        # logging.info(f"[DEBUG] normalized entropies: {normalized_entropies}")

        with torch.no_grad():
            if self.adapted_model_key == "tasc":
                if self.use_unk_classifier:
                    output_norm, _ = self.norm_model(input_imgs_origin[known_mask]) # predictions before update
                else:
                    output_norm, _ = self.norm_model(input_imgs_origin) # predictions before update
            else:
                if self.use_unk_classifier:
                    output_norm = self.norm_model(input_imgs_origin[known_mask]) # predictions before update
                else:
                    output_norm = self.norm_model(input_imgs_origin) # predictions before update
        
        # keep predictions that match the original model
        if self.consistency_filtering:
            logging.info(f"[DEBUG] applying consistency filtering")
            logging.info(f"[DEBUG] output_norm.shape[0]: {output_norm.shape[0]}")
            logging.info(f"[DEBUG] output_origin_retraining.shape[0]: {output_origin_retraining.shape[0]}")
            filter_ids_0 = torch.where(output_origin_retraining.max(dim=1)[1] == output_norm.max(dim=1)[1])
        else:
            # keep all predictions
            batch_size = output_origin_retraining.shape[0]
            filter_ids_0 = torch.arange(batch_size, device=output_origin_retraining.device)

        # SAFETY CHECK:
        logger.info(f"[DEBUG] filter_ids_0: {filter_ids_0} (expected to be non-empty)")

        outputs.append(output_origin) # we keep all predictions for original images
        
        # obtain model predictions for augmented images
        for i in range(1, len(input_imgs)):
            input_imgs_aug = input_imgs[i]
            with torch.no_grad():
                if self.adapted_model_key == "tasc":
                    outputs.append(self.model(input_imgs_aug)[0])
                else:
                    outputs.append(self.model(input_imgs_aug))            
            del input_imgs_aug # to free-up memory

        output = torch.stack(outputs, dim=0)
        
        # SAFETY CHECK:
        logger.info(f"[DEBUG] before averaging: output.shape = {output.shape} (expected [K+1, B, num_classes])")
        output = torch.mean(output, dim=0)
        logger.info(f"[DEBUG] after averaging: output.shape = {output.shape} (expected [K+1, B, num_classes])")
        
        # compute entropy and find low-entropy samples
        if self.use_unk_classifier:
            logging.info(f"[DEBUG] output[known_mask].shape[0]: {output[known_mask].shape[0]}")
            entropys = entropy(output[known_mask])[filter_ids_0]
            logging.info(f"[DEBUG] entropys after filtering: {entropys.shape[0]}")
        else:
            logging.info(f"[DEBUG] output.shape[0]: {output.shape[0]}")
            entropys = entropy(output)[filter_ids_0]
            logging.info(f"[DEBUG] entropys after filtering: {entropys.shape[0]}")
        filter_ids = torch.where(entropys < self.margin)

        # SAFETY CHECK:
        # logging.info(f"[DEBUG] entropys (averaged) = {entropys.tolist()}")
        logger.info(f"[DEBUG] filter_ids: {filter_ids} (expected to be non-empty)")
        
        # keep predictions for low-entropy samples
        if self.use_unk_classifier:
            input_imgs_append = input_imgs_origin[known_mask][filter_ids_0][filter_ids]
            self.mem.append(input_imgs_append, output_origin[known_mask].max(dim=1)[1][filter_ids_0][filter_ids])
        else:
            input_imgs_append = input_imgs_origin[filter_ids_0][filter_ids]
            self.mem.append(input_imgs_append, output_origin.max(dim=1)[1][filter_ids_0][filter_ids])
        logger.info(f"[DEBUG] memory size after updt = {len(self.mem)}")

        logger.info("[DEBUG] ---end: update_memory---")
        
        logging.info(f"[DEBUG]: final output shape: {output.shape[0]}")
        if self.adapted_model_key == "tasc":
            return output, unknown_scores_dict
        else:
            return output

    @torch.enable_grad()
    def adapt(self):
        """
        Adapt the model on the samples in memory using SAM optimizer
        """
        
        data = self.mem.get_data()
        self.optimizer.zero_grad()

        if len(data) > 1: # fails if only 1 sample in memory

            ## first time: computing losses at the current model state
            # obtaining model predictions
            if self.adapted_model_key == "tasc":
                output_1, _ = self.model(data)
            else:
                output_1 = self.model(data)
            entropys = entropy(output_1)

            # coeff = 1 / (torch.exp(entropys.clone().detach() - self.margin))
            # obtaining weighted entropy
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()

            # minimizing weighted entropy
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            ## second time: computing losses at the updated model state
            # obtaining model predictions
            if self.adapted_model_key == "tasc":
                output_1, _ = self.model(data)
            else:
                output_1 = self.model(data)
            entropys = entropy(output_1)

            # obtaining weighted entropy
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean()

            # minimizing weighted entropy
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()

class GmmBaAdaptationModule(BaseModule):
    def __init__(self, model, model_name, shared_class_num, source_private_class_num, lr=1e-2, red_feature_dim=64, p_reject=0.5, N_init=30,
                 augmentation=True, lam=1, temperature=0.1, device="cuda", use_unk_classifier=None):
        super(GmmBaAdaptationModule, self).__init__(model, model_name, shared_class_num, source_private_class_num, lr)

        logger = logging.getLogger(__name__)
        logger.info(f"GmmBaAdaptationModule.device = {device}")

        if self.model_name == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr

        # ---------- Dataset information ----------
        self.source_class_num = source_private_class_num + shared_class_num
        self.known_classes_num = source_private_class_num + shared_class_num
        self.num_classes = source_private_class_num + shared_class_num
        self.use_unk_classifier = use_unk_classifier

        # Additional feature reduction model
        self.feature_reduction = nn.Sequential(nn.Linear(self.feature_dim, red_feature_dim)).to(device)
        setattr(self.model, "feature_reduction", self.feature_reduction)

        # ---------- GMM ----------
        self.gmm = GaussianMixtureModel(self.source_class_num)

        # ---------- Unknown mask ----------
        self.mask = mask(0.5 - p_reject / 2, 0.5 + p_reject / 2, N_init)

        # ---------- Further initializations ----------
        self.tta_transform = get_tta_transforms_gmm()
        self.augmentation = augmentation
        self.temperature = temperature
        self.lam = lam

        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        """
        Collects parameters and defines optimizer
        """
        
        logger = logging.getLogger(__name__)
        logger.info("[DEBUG] ---start: collect_optimizers---")

        # collect parameter groups
        params_group = []

        # lora params (by name match)
        model_uses_lora = False
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                params_group.append({"params": param, "lr": self.lr * 0.1})
                logger.info(f"[lora] {name}")
                
                if not model_uses_lora:
                    model_uses_lora = True
        
        if model_uses_lora:
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for name, param in m.named_parameters():
                        if name in ["weight", "bias"]: # weight is scale, bias is shift
                            params_group.append({"params": param, "lr": self.lr})
                            logger.info(f"[norm] {name}")
        
        if not model_uses_lora:
            for name, param in self.model.named_parameters():
                # backbone
                if "backbone" in name.lower() and param.requires_grad:
                    params_group.append({"params": param, "lr": self.lr * 0.1})
                    logger.info(f"[backbone] {name}")

                # feature extractor
                if "feature_extractor" in name.lower() and param.requires_grad:
                    params_group.append({"params": param, "lr": self.lr})
                    logger.info(f"[feature_extractor] {name}")

                # classifier
                if "classifier" in name.lower() and param.requires_grad:
                    params_group.append({"params": param, "lr": self.lr})
                    logger.info(f"[classifier] {name}")

        for name, param in self.model.named_parameters():
            # feature reduction
            if "feature_reduction" in name.lower() and param.requires_grad:
                params_group.append({"params": param, "lr": self.lr})
                logger.info(f"[feature_reduction] {name}")

        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        logger.info("[DEBUG] ---end: collect_optimizers---")
        return optimizer

    @torch.enable_grad()
    def training_step(self, x, device="cuda"):
        logger = logging.getLogger(__name__)
        logger.info("[DEBUG] ---start: training_step---")

        # ----------- Open-World Test-time Training ------------
        self.model.train()

        with autocast():
            if self.model_name == "tasc":
                feat_ext, logit_hat, unknown_scores_dict = self.model(x, return_feats=True)
                
                # storing original features and logits
                feat_ext_orig = feat_ext.clone().detach()
                logit_hat_orig = logit_hat.clone().detach()
                
                if self.use_unk_classifier: # filtering out unreliable samples according to model
                    known_scores = -unknown_scores_dict["UniMS"]
                    known_scores = (known_scores - known_scores.min()) / (known_scores.max() - known_scores.min())

                    thr = self.thr_curr if self.thr_curr is not None else 0
                    logging.info(f"[DEBUG] thr for known/unknown separation: {thr}")
                    known_mask = known_scores >= thr       
                    logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples")
                    feat_ext, logit_hat = feat_ext[known_mask], logit_hat[known_mask]
                else:
                    known_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

                feat_ext_aug, logit_hat_aug, unknown_scores_dict_aug = self.model(self.tta_transform(x[known_mask]), return_feats=True)
                y_hat = F.softmax(logit_hat, dim=1)
                y_hat_aug = F.softmax(logit_hat_aug, dim=1)

                # obtaining original predictions for all samples
                y_hat_orig = F.softmax(logit_hat_orig, dim=1)

            else:
                feat_ext, logit_hat = self.model(x, return_feats=True)

                # storing original features and logits
                feat_ext_orig = feat_ext.clone().detach()
                logit_hat_orig = logit_hat.clone().detach()

                if self.use_unk_classifier:
                    entropies = entropy(F.softmax(logit_hat, dim=1))
                    known_mask = entropies < 0.55 # threshold used in LEAD and GLC
                    logging.info(f"[DEBUG] adapting with {torch.sum(known_mask).item()} reliable samples out of {batch_size} samples")
                    feat_ext, logit_hat = feat_ext[known_mask], logit_hat[known_mask]
                else:
                    known_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

                feat_ext_aug, logit_hat_aug = self.model(self.tta_transform(x[known_mask]), return_feats=True)
                y_hat = F.softmax(logit_hat, dim=1)
                y_hat_aug = F.softmax(logit_hat_aug, dim=1)

                # obtaining original predictions for all samples
                y_hat_orig = F.softmax(logit_hat_orig, dim=1)

            with torch.no_grad():
                feat_ext = self.feature_reduction(feat_ext)
                feat_ext_aug = self.feature_reduction(feat_ext_aug)
                # Update the GMM
                y_hat_clone_detached = y_hat.clone().detach()
                self.gmm.soft_update(feat_ext, y_hat_clone_detached)
                
                _, pseudo_labels, likelihood = self.gmm.get_labels(feat_ext)
                pseudo_labels = pseudo_labels.to(device)
                likelihood = likelihood.to(device)
                
            # ---------- Generate a mask and monitor the result ----------
            known_mask, unknown_mask, rejection_mask = self.mask.calculate_mask(likelihood)
            
            known_mask = known_mask.to(device)
            unknown_mask = unknown_mask.to(device)
            rejection_mask = rejection_mask.to(device)

            # Assign unknown pseudo-labels
            pseudo_labels[unknown_mask] = self.source_class_num
            
            # ---------- Enable OPDA for predictions ----------
            _, preds = torch.max(y_hat_clone_detached, dim=1)
            unknown_threshold = (self.mask.tau_low + self.mask.tau_high) / 2
            
            entropy_values = calculate_entropy_from_likelihood(likelihood)
            output_mask = torch.zeros_like(entropy_values, dtype=torch.bool)
            output_mask[entropy_values >= unknown_threshold] = True
            preds[output_mask] = self.source_class_num
            self.last_preds_with_unknown = preds.detach().clone() # NOTE: used for metrics calculation in BaseModule
            
            # ---------- Calculate the loss -----------
            # ---------- Contrastive loss -----------
            feat_ext = feat_ext.to(device)
            feat_ext_aug = feat_ext_aug.to(device)
            if self.augmentation:
                feat_total = torch.cat([feat_ext, feat_ext_aug], dim=0)
            else:
                feat_total = feat_ext
            mu = self.gmm.mu.to(device)
            # Calculate all cosine similarities between features (embeddings)
            cos_feat_feat = torch.exp(calculate_cosine_similarity(feat_total, feat_total) / self.temperature)
            # Calculate all cosine similarities between features (embeddings) and GMM means
            cos_feat_mu = torch.exp(calculate_cosine_similarity(mu, feat_total) / self.temperature)
            
            # Minimize distance between known features and their corresponding mean
            # Maximize distance between known/unknown features and the mean of different classes
            divisor = torch.sum(cos_feat_mu, dim=0)
            logarithmus = torch.log(torch.divide(cos_feat_mu, divisor.unsqueeze(0)))
            if self.augmentation:
                known_mask_rep = known_mask.repeat(2)
                pseudo_labels_rep = pseudo_labels.repeat(2)
            else:
                known_mask_rep = known_mask
                pseudo_labels_rep = pseudo_labels
            used = torch.gather(logarithmus[known_mask_rep], 1, pseudo_labels_rep[known_mask_rep].view(-1, 1))
            L_mu_feat = torch.sum(torch.sum(used, dim=0))

            # Minimize distance between known features of the same class
            # Maximize distance between known/unknown features of different classes
            divisor = torch.sum(cos_feat_feat, dim=0)
            logarithmus = torch.log(torch.divide(cos_feat_feat, divisor.unsqueeze(0)))
            # Calculate the equality between elements of pseudo_label along both axes
            pseudo_label_rep_expanded = pseudo_labels_rep.unsqueeze(1)
            mask = pseudo_label_rep_expanded == pseudo_labels_rep
            used = torch.zeros_like(logarithmus)
            used[mask] = logarithmus[mask.bool()]

            L_feat_feat = torch.sum(torch.sum(used[known_mask_rep, known_mask_rep], dim=0))
            L_con = L_mu_feat + L_feat_feat

            # ---------- KL-Divergence loss ----------
            # Maximize divergence between uniform distribution and models output of known classes
            likelihood = y_hat[known_mask,:]
            true_dist = torch.ones_like(likelihood) / 1e3
            kl_known = - torch.sum(calculate_kld(likelihood, true_dist))

            # Minimize divergence between uniform distribution and models output of unknown classes
            likelihood = y_hat[unknown_mask,:]
            true_dist = torch.ones_like(likelihood) / 1e3
            kl_unknown = torch.sum(calculate_kld(likelihood, true_dist))

            L_kl = kl_known + kl_unknown

            self.loss = L_con + self.lam * L_kl
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # log into progress bar
        logger.info(f"[DEBUG] train_loss: {self.loss}")
        logger.info("[DEBUG] ---end: training_step---")
        if self.model_name == "tasc":
            return y_hat_orig, feat_ext_orig, unknown_scores_dict
        else:
            return y_hat_orig, feat_ext_orig        

    def forward(self, input_imgs):
        logger = logging.getLogger(__name__)
        batch_size = int(input_imgs.size(0))
        batch_shape = input_imgs.shape
        logging.info(f"[DEBUG] batch_size: {batch_size}")
        logging.info(f"[DEBUG] batch_shape: {batch_shape}")

        if batch_size > 1:
            if self.model_name == "tasc":
                y_hat, feat_ext, unknown_scores_dict = self.training_step(input_imgs) 
                logging.info(f"[DEBUG]: final output shape: {y_hat.shape[0]}")
                return y_hat, unknown_scores_dict
            else:
                y_hat, feat_ext = self.training_step(input_imgs) 
                logging.info(f"[DEBUG]: final output shape: {y_hat.shape[0]}")
                return y_hat
        else:
            logging.info(f"[DEBUG] no adaptation performed, batch size: {batch_size}")
            self.model.eval()
            if self.model_name == "tasc":
                with torch.no_grad():
                    output, unknown_scores_dict = self.model(input_imgs) # input_imgs[0] is the original image
                    logging.info(f"[DEBUG]: final output shape: {output.shape[0]}")
                return output, unknown_scores_dict
            else:
                with torch.no_grad():
                    output = self.model(input_imgs) # input_imgs[0] is the original image
                    logging.info(f"[DEBUG]: final output shape: {output.shape[0]}")
                return output
                
class UnidaTTA(nn.Module):
    """
    Custom TTA class that uses a KL divergence loss to avoid steering to far
    from the model's training and consistency filtering to retrain w/ most
    reliable samples.

    NOTE: Two possible CF boosters using Neyman-Pearson:

    1. balanced np: class-balanced memory of good and bad samples 
    2. dirty np: balanced memory + current batch samples with neyman-pearson filtering
    """
    
    def __init__(self, model, model_key, checkpoint_path, backbone_source, optimizer,
                 target_classnames_gsearch=None, cluster_params_path=None, source_classnames=None, 
                 device="cuda", consistency_filtering=None, kl_weight=0, alpha=0.05,
                 type_cf_booster=None, memory_reset=None, risk_memory_size=None):
        super(UnidaTTA, self).__init__()
        
        # loading model and optimizer
        self.device = device
        self.adapted_model_key = model_key
        self.optimizer = optimizer
        self.load_base_model(checkpoint_path, model, backbone_source, target_classnames_gsearch, 
                             cluster_params_path, source_classnames)
        
        # work-in-progress parameters
        self.consistency_filtering = consistency_filtering
        self.type_cf_booster = type_cf_booster # "balanced" or "dirty"
        self.memory_reset = memory_reset # if True, resets sample memory after each batch

        # setting up TTA parameters
        self.kl_weight = kl_weight
        self.alpha = alpha # significance level for neyman-pearson filtering
        
        logging.info(f"[DEBUG] UnidaTTA model configured with num_class = {self.num_classes}")

        # initializing memories
        self.mem = RBM(64, self.num_classes)
        self.rsm = RBM(risk_memory_size, self.num_classes) # reliable samples score memory
        self.usm = RBM(risk_memory_size, self.num_classes) # unreliable samples score memory
        
        # collecting parameters to adapt
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        if self.num_classes == 1000:
            self.max_iter = 750
        else:
            self.max_iter = 150
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iter)

    def load_base_model(self, checkpoint_path, model, backbone_source, target_classnames_gsearch, 
                        cluster_params_path, source_classnames):
        """
        Obtains model and optimizer states for reset
        """

        self.model = self.configure_model(model)
        self.num_classes = self.model.num_classes

        if self.adapted_model_key == "tasc":
            self.estimated_shared = self.model.estimated_shared
            self.num_clusters = self.model.num_clusters
            self.thr_curr = self.model.thr_curr
            self.norm_model = load_unida_model(self.adapted_model_key, checkpoint_path, num_classes=self.num_classes, backbone_source=backbone_source,
                                               target_classnames_gsearch=target_classnames_gsearch, cluster_params_path=cluster_params_path,
                                               source_classnames=source_classnames).train()
        else:
            self.norm_model = load_unida_model(self.adapted_model_key, checkpoint_path, num_classes=self.num_classes, backbone_source=backbone_source).train()
        self.norm_model.requires_grad_(False)
        
    @staticmethod
    def configure_model(model):
        model.train()
        model.requires_grad_(False)

        logger = logging.getLogger(__name__)
        logging.info("[DEBUG] ---start: configure_model---")

        # configure norm for eata updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

                # SAFETY CHECK:
                logging.info(f"[DEBUG] enabling grad for {m} (BatchNorm2d) and forcing batch statistics")
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

                # SAFETY CHECK:
                logging.info(f"[DEBUG] enabling grad for {m} (LayerNorm) and forcing batch statistics")
        
        logging.info("[DEBUG] ---end: configure_model---")
        return model

    @staticmethod
    def collect_params(model):
        """
        Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []
        logger = logging.getLogger(__name__)

        logging.info("[DEBUG] ---start: collect_params---")
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if "layer4" in nm:
                continue
            if "conv5_x" in nm:
                continue
            if "blocks.9" in nm:
                continue
            if "blocks.10" in nm:
                continue
            if "blocks.11" in nm:
                continue
            if "norm." in nm:
                continue
            if nm in ["norm"]:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        # SAFETY CHECK:
        # logger.info(f"[DEBUG] collected {len(params)} parameters for STAMP adaptation: {names}")
        logging.info("[DEBUG] ---end: collect_params---")

        return params, names

    @staticmethod
    def kl_divergence(src_probs, adp_probs):
        """
        Computes the KL forward divergence between the source and adapted logits
        (inspired by AntiCF)
        """

        return (src_probs * (torch.log(src_probs) - torch.log(adp_probs))).sum(dim=1)
  
    def forward(self, input_imgs):
        logger = logging.getLogger(__name__)
        
        if self.adapted_model_key == "tasc":
            output, unknown_scores_dict, output_norm = self.update_memory(input_imgs) 
            if len(self.mem) != 0:
                logger.info(f"[DEBUG] memory size before adapt = {len(self.mem)}")
                self.adapt(output, output_norm)
                if self.memory_reset: self.mem.reset()  
                logger.info(f"[DEBUG] memory size after adapt = {len(self.mem)}")
            return output, unknown_scores_dict
        else:
            output, output_norm = self.update_memory(input_imgs) 
            if len(self.mem) != 0:
                logger.info(f"[DEBUG] memory size before adapt = {len(self.mem)}")
                self.adapt(output, output_norm)
                if self.memory_reset: self.mem.reset()  
                logger.info(f"[DEBUG] memory size after adapt = {len(self.mem)}")
            return output

    def update_memory(self, input_imgs):
        """
        Selects low-entropy samples from the input images and appends them to the memory
        """

        input_imgs_origin = input_imgs[0] # x is a list of augmented images, first is the original image

        logger = logging.getLogger(__name__)
        logger.info("[DEBUG] ---start: update_memory---")
        outputs = []
        self.model.train() # NOTE: this seems to be implemented for safety, but not sure if needed

        with torch.no_grad():
            if self.adapted_model_key == "tasc":
                output_origin, unknown_scores_dict = self.model(input_imgs_origin) # predictions after update
            else:
                output_origin = self.model(input_imgs_origin) # predictions after update
        
        # SAFETY CHECK
        with torch.no_grad():
            if self.adapted_model_key == "tasc":
                output_norm, _ = self.norm_model(input_imgs_origin) # predictions before update
            else:
                output_norm = self.norm_model(input_imgs_origin) # predictions before update
        
        outputs.append(output_origin)
        
        # obtain model predictions for augmented images
        for i in range(1, len(input_imgs)):
            input_imgs_aug = input_imgs[i]
            with torch.no_grad():
                if self.adapted_model_key == "tasc":
                    outputs.append(self.model(input_imgs_aug)[0])
                else:
                    outputs.append(self.model(input_imgs_aug))            
            del input_imgs_aug # to free-up memory

        output = torch.stack(outputs, dim=0)
        
        # SAFETY CHECK:
        logger.info(f"[DEBUG] before averaging: output.shape = {output.shape} (expected [K+1, B, num_classes])")
        output = torch.mean(output, dim=0)
        logger.info(f"[DEBUG] after averaging: output.shape = {output.shape} (expected [K+1, B, num_classes])")

        # keep predictions that match the original model
        if self.consistency_filtering:
            filter_ids_cf = torch.where(output.max(dim=1)[1] == output_norm.max(dim=1)[1])
            filter_ids_no_cf = torch.where(output.max(dim=1)[1] != output_norm.max(dim=1)[1])
            logger.info(f"[DEBUG] filter_ids_no_cf: {filter_ids_no_cf[0].size(0)} (before update)")
        else:
            batch_size = output_origin.shape[0] # keep all predictions
            filter_ids_cf = torch.arange(batch_size, device=output_origin.device)

        # SAFETY CHECK:
        logger.info(f"[DEBUG] filter_ids_cf: {filter_ids_cf[0].size(0)} (before update)")
        
        # Neyman-Pearson filtering
        np_min_samples_no_cf = math.ceil(1/self.alpha)
        balanced_filtering_condition = (self.type_cf_booster == "balanced" and len(self.usm) >= np_min_samples_no_cf)
        dirty_filtering_condition = (self.type_cf_booster == "dirty" and len(self.usm) + filter_ids_no_cf[0].size(0) >= np_min_samples_no_cf)
        if self.consistency_filtering and self.type_cf_booster in ["balanced", "dirty"] and (balanced_filtering_condition or dirty_filtering_condition):
            logger.info("[DEBUG] ---start: Neyman-Pearson filtering---")
            if balanced_filtering_condition:
                logger.info("[DEBUG] using balanced Neyman-Pearson filtering")
                unreliable_sample_scores = self.usm.get_data()
                reliable_sample_scores = self.rsm.get_data()
            
            else: # dirty_filtering_condition
                logger.info("[DEBUG] using dirty Neyman-Pearson filtering")
                # obtain risk scores from rsm
                scores_usm = self.usm.get_data()
                scores_rsm = self.rsm.get_data()

                logger.info(f"[DEBUG] scores_usm.shape = {scores_usm.shape}")
                logger.info(f"[DEBUG] scores_rsm.shape = {scores_rsm.shape}")

                # obtain risk scores from current samples
                unreliable_sample_scores_no_rsm = output.max(dim=1)[0][filter_ids_no_cf]
                reliable_sample_scores_no_rsm = output.max(dim=1)[0][filter_ids_cf]

                logger.info(f"[DEBUG] unreliable_sample_scores_no_rsm.shape = {unreliable_sample_scores_no_rsm.shape}")
                logger.info(f"[DEBUG] reliable_sample_scores_no_rsm.shape = {reliable_sample_scores_no_rsm.shape}")

                # stack risk scores
                unreliable_sample_scores = torch.cat((scores_usm, unreliable_sample_scores_no_rsm), dim=0)
                reliable_sample_scores = torch.cat((scores_rsm, reliable_sample_scores_no_rsm), dim=0)

            logger.info(f"[DEBUG] unreliable_sample_scores.shape = {unreliable_sample_scores.shape}")
            logger.info(f"[DEBUG] reliable_sample_scores.shape = {reliable_sample_scores.shape}")
            
            # obtain Neyman-Pearson threshold
            np_threshold_batch, density_unreliable, density_reliable = np_threshold(unreliable_sample_scores, reliable_sample_scores)
            logger.info(f"[DEBUG] np_threshold = {np_threshold_batch}")

            # finding samples that pass Neyman-Pearson filtering
            density_unreliable = vectorize_density(density_unreliable)
            density_reliable = vectorize_density(density_reliable)
            np_mask = np_optimal_classifier(output.max(dim=1)[0], np_threshold_batch, density_unreliable, density_reliable)

            logger.info(f"[DEBUG] np_mask.shape = {np_mask.shape} (expected [B])")

            filter_ids_cf = torch.where(np_mask == 1)[0]
            filter_ids_no_cf = torch.where(np_mask == 0)[0]

            logger.info(f"[DEBUG] updated filter_ids_cf: {filter_ids_cf.size(0)} (expected to be non-empty)")
            logger.info(f"[DEBUG] updated filter_ids_no_cf: {filter_ids_no_cf.size(0)} (expected to be non-empty)")
        else:
            logger.info(f"[DEBUG] no Neyman-Pearson filtering for this batch, type_cf_booster {self.type_cf_booster}, balanced_filtering_condition {balanced_filtering_condition}, dirty_filtering_condition {dirty_filtering_condition}")
            pass

        # updating memory with filtered samples
        input_imgs_append = input_imgs_origin[filter_ids_cf]
        self.mem.append(input_imgs_append, output.max(dim=1)[1][filter_ids_cf])

        # updating risk score memories 
        self.rsm.append(output.max(dim=1)[0][filter_ids_cf], output.max(dim=1)[1][filter_ids_cf])
        self.usm.append(output.max(dim=1)[0][filter_ids_no_cf], output.max(dim=1)[1][filter_ids_no_cf])

        logger.info(f"[DEBUG] RBM size after updt = {len(self.mem)}")
        logger.info(f"[DEBUG] RSM size after updt = {len(self.rsm)}")
        logger.info(f"[DEBUG] USM size after updt = {len(self.usm)}")

        logger.info("[DEBUG] ---end: update_memory---")
        
        if self.adapted_model_key == "tasc":
            return output, unknown_scores_dict, output_norm
        else:
            return output, output_norm

    @torch.enable_grad()
    def adapt(self, output, output_norm):
        """
        Adapt the model on the samples in memory using SAM optimizer
        """
        
        data = self.mem.get_data()
        self.optimizer.zero_grad()

        if len(data) > 1: # fails if only 1 sample in memory

            ## kl regularization
            if self.kl_weight > 0:
                kl_loss = self.kl_weight * (self.kl_divergence(output_norm, output)).mean()
            else:
                kl_loss = 0

            ## first time: computing losses at the current model state
            # obtaining model predictions
            if self.adapted_model_key == "tasc":
                output_1, _ = self.model(data)
            else:
                output_1 = self.model(data)
            entropys = entropy(output_1)

            # coeff = 1 / (torch.exp(entropys.clone().detach() - self.margin))
            # obtaining weighted entropy
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean() + kl_loss

            # minimizing weighted entropy
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            ## second time: computing losses at the updated model state
            # obtaining model predictions
            if self.adapted_model_key == "tasc":
                output_1, _ = self.model(data)
            else:
                output_1 = self.model(data)
            entropys = entropy(output_1)

            # obtaining weighted entropy
            inv_entropy = 1 / torch.exp(entropys)
            coeff = inv_entropy / inv_entropy.sum() * 64
            entropys = entropys.mul(coeff)
            loss = entropys.mean() + kl_loss

            # minimizing weighted entropy
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
            self.scheduler.step()

    def reset(self):
        """
        Resets parameters of the model and optimizer to their initial state
        NOTE: used to reset after implemented on a corruption
        NOTE: unused in our implementation
        """

        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.mem.reset()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_iter)
