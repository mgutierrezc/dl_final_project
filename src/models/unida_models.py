import clip
import json
import numpy as np
import os, torch
from math import log
import torch.nn as nn
import torch.nn.functional as F
import logging, random
from torchvision import models
from copy import deepcopy
from .templates import get_templates
from ..utils import remap_state_dict

backbones = {"resnet18":models.resnet18, "resnet34":models.resnet34, 
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

CLIP_MODELS = clip.available_models() 

# CODE CITATIONS:
# the Unida models are based on the code from https://github.com/ispc-lab/LEAD and https://github.com/ispc-lab/GLC

#### helpers ####

def init_weights(m):
    """
    Initialize weights for layers in models
    NOTE: Mainly used for transformations of Embedding class
    """

    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def load_unida_model(model_name, checkpoint_path, **kwargs):
    model_mapping = {
        "glc": GLC,
        "lead": LEAD
    }

    try:
        model_class = model_mapping[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name '{model_name}'. Available models: {list(model_mapping.keys())}")

    return model_class(checkpoint_path, **kwargs)

#### feature extraction classes ####

class ResBase(nn.Module):
    """
    Backbone class for ResNet architectures
    NOTE: UniDA models take a ResNet50 as the feature extractor backbone
    """
    def __init__(self, res_name="resnet50"):
        super().__init__() # calls the parent class constructor
        
        model_resnet = backbones[res_name](pretrained=True) 

        # setting up each layer
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        """
        Obtains base features from the input image tensor x
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class Embedding(nn.Module):
    """
    Embedding layer for feature bottleneck
    NOTE: This layer reduces the feature dimension to a specified 
    embedding dimension (changes by model)
    """
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
        super(Embedding, self).__init__()

        # initializing layers and parameters
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        """
        Transforms the input feature tensor x into
        embedding of fixed size.
        NOTE: Should run after the backbone feature extraction.
        """

        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class Classifier(nn.Module):
    """
    Weight normalized linear layer
    Used in Source-Free UniDA classifiers
    """

    def __init__(self, embed_dim, class_num, type="linear"):
        super().__init__()
        self.type = type

        if type == "wn": # weight‑normalization version
            self.fc = nn.utils.weight_norm(
                nn.Linear(embed_dim, class_num), name="weight"
            )
        else: 
            self.fc = nn.Linear(embed_dim, class_num)
        
        # self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(x)


#### unida models ####

class UnidaModel(nn.Module):
    """
    Base class for all UniDA models
    """

    def __init__(self, checkpoint_path, backbone_arch="resnet50", backbone_source="download", 
                 backbone_checkpoint_path=None, device="cuda", num_classes=11, 
                 needs_remapping=False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path # None if training 

        # backbone parameters
        self.backbone_arch = backbone_arch
        self.backbone_source = backbone_source
        self.backbone_checkpoint_path = backbone_checkpoint_path
        
        # loading backbone (overwrite function on child class if loading is needed)
        if backbone_source == "download": 
            self.backbone = self.load_backbone()
        self.needs_remapping = needs_remapping # whether checkpoint keys need remapping 
    
    def load_backbone(self):
        """
        Loads the backbone ('download' or 'pth')
        """

        if self.backbone_source == "download":
            backbone = ResBase(self.backbone_arch) 
        elif self.backbone_source == "pth":
            if self.backbone_checkpoint_path is None:
                raise ValueError("backbone_checkpoint_path must be provided when backbone_source is 'pth'.")
            backbone = torch.load(self.backbone_checkpoint_path, map_location=self.device)
        else:
            raise ValueError("Invalid backbone_source. Choose 'download' or 'pth'.")
        
        backbone.to(self.device)
        backbone.eval()
        return backbone
    
    def get_input_features(self, input_imgs):
        """
        Obtains features from backbone for input image
        NOTE: this should be overridden in derived classes
        """
        backbone_features = self.backbone(input_imgs)
        return backbone_features

    @torch.no_grad()
    def forward(self, backbone_features):
        """
        Default forward pass
        """
        return self.model(backbone_features)
    
    
class GLC(UnidaModel):
    """
    GLC (Source-Free architecture)
    """

    def __init__(self, checkpoint_path, n_features=256, num_classes=11,
                 backbone_arch="resnet50", backbone_source="download", 
                 backbone_checkpoint_path=None, device="cuda", 
                 needs_remapping=False):
        
        # pass the necessary parameters to the base class
        super().__init__(checkpoint_path, backbone_arch, backbone_source, 
                         backbone_checkpoint_path, device, num_classes,
                         needs_remapping)
        
        # setting up embedding extractor
        self.embeddings_dim = n_features
        self.embedding_layer = Embedding(
            feature_dim=self.backbone.backbone_feat_dim,
            embed_dim=self.embeddings_dim,
            type="bn"  # BatchNorm type for embedding
        )

    def get_input_features(self, input_imgs):
        """
        Obtains features from backbone for input image
        NOTE: this should be overridden in derived classes
        """

        backbone_features = self.backbone(input_imgs)
        embeddings = self.embedding_layer(backbone_features)
        return embeddings 

    @torch.no_grad()
    def forward(self, input_imgs):
        """
        Obtains predictions from the model for the input images
        in all UniDA cases
        """

        embeddings = self.get_input_features(input_imgs)
        logits = self.model(embeddings)
        probabilities = torch.softmax(logits, dim=1)
        return logits, probabilities


class LEAD(UnidaModel):
    """
    LEAD (Source-Free architecture)
    """

    def __init__(self, checkpoint_path, n_features=256, num_classes=11, 
                 backbone_arch="resnet50", backbone_source="download", 
                 backbone_checkpoint_path=None, device="cuda", 
                 needs_remapping=True):
        
        # pass the necessary parameters to the base class
        super().__init__(checkpoint_path, backbone_arch, backbone_source, 
                         backbone_checkpoint_path, device, num_classes,
                         needs_remapping)
        
        # TODO: use n_features for initialization of embedding layer when no pth available
        self.load_layers() # initializes or loads layers
        self.to(self.device) # move model to device

    def get_input_features(self, input_imgs):
        """
        Obtains features from backbone for input image
        NOTE: this should be overridden in derived classes
        """

        backbone_features = self.backbone(input_imgs)
        embeddings = self.embedding_layer(backbone_features)
        return embeddings 
    
    def load_layers(self):
        """
        Initializes weights and biases or loads the model weights from checkpoint
        """

        if self.checkpoint_path != None:
            
            # load the checkpoint dictionary
            model_state_dict = torch.load(self.checkpoint_path, map_location=self.device)["model_state_dict"]
            if self.needs_remapping:
                model_state_dict = remap_state_dict(model_state_dict)

            # obtaining weights for all layers
            backbone_weights = {}
            embedding_weights = {}
            classifier_weights = {}
            for key, value in model_state_dict.items():
                if key.startswith("backbone."):
                    backbone_weights[key[len("backbone."):]] = value
                elif key.startswith("embedding_layer."):
                    # initializing layer without "embedding_layer." prefix
                    embedding_weights[key[len("embedding_layer."):]] = value
                else:
                    # initializing layer without "model." prefix
                    classifier_weights[key[len("model."):]] = value
            
            # initializing backbone layer
            self.backbone = ResBase(self.backbone_arch).to(self.device)
            self.backbone.load_state_dict(backbone_weights, strict=True)
            print("loaded backbone layer")

            # initializing embedding layer
            self.embedding_layer = Embedding(
                feature_dim=self.backbone.backbone_feat_dim,
                embed_dim=embedding_weights["bottleneck.weight"].shape[0], # inferring dimensions
                type="bn").to(self.device)
            self.embedding_layer.load_state_dict(embedding_weights, strict=True)
            print("loaded embedding layer")

            self.feature_dim = embedding_weights["bottleneck.weight"].shape[0]
            self.output_dim = embedding_weights["bottleneck.weight"].shape[0]

            # initializing clasifier layer
            self.classifier = Classifier(
                embed_dim=self.embedding_layer.bottleneck.out_features,
                class_num=self.num_classes,
                type="wn").to(self.device)
            self.classifier.load_state_dict(classifier_weights, strict=True)
            print("loaded classifier layer")

            self.model = self.classifier # aliasing classifier as model

    # @torch.no_grad()
    def forward(self, input_imgs, return_feats=False):
        """
        Obtains predictions from the model for the input images
        in all UniDA cases
        """

        embeddings = self.get_input_features(input_imgs)
        logits = self.model(embeddings)

        if return_feats:
            return embeddings, logits
        else:
            probabilities = torch.softmax(logits, dim=1)
            return probabilities


if __name__ == "__main__":
    print("Module with UniDA models")