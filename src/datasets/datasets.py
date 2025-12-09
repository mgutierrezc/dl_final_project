import logging
import os, torch
from tqdm import tqdm
from numpy import random
from PIL import Image
import PIL
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda

# CODE CITATIONS:
# the classes for the TTA transformations come from https://github.com/yuyongcan/STAMP/blob/master/src/data/augmentations/transforms_cotta.py#L7
# the classes for Corruptions come from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py

#### helpers ####

def preprocess_images_eval(resize_size=256, crop_size=224):
    """
    Preprocess images for evaluation
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def read_datalist(file_path):
    """
    Reads the datalist from a text file where each line consists of an image path and a label
    """

    return open(file_path, "r").readlines()

def get_tta_transforms(dataset_name):
    """
    Get the TTA transforms for a given dataset
    """

    if "cifar" in dataset_name:
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip()])
    else:
        transform = transforms.Compose([transforms.RandomCrop(224, padding=4),
                                       transforms.RandomHorizontalFlip()])
    return transform

def get_tta_transforms_gmm(gaussian_std: float = 0.005, soft=False, padding_mode="edge", cotta_augs=True):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=0
        )
    ]
    if cotta_augs:
        tta_transforms += [transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
                           transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           GaussianNoise(0, gaussian_std),
                           Clip(0.0, 1.0)]
    else:
        tta_transforms += [transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           Clip(0.0, 1.0)]

    return transforms.Compose(tta_transforms)

#### dataset classes ####

class GaussianNoise(torch.nn.Module):
    """
    Add Gaussian noise to an image
    """

    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Clip(torch.nn.Module):
    """
    Clip the pixel values of an image to a specified range
    """

    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)

class ColorJitterPro(ColorJitter):
    """
    Randomly change the brightness, contrast, saturation, and gamma correction of an image
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """
        Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string

class UnidaDataset(Dataset):
    """
    A dataset class for UniDA (Universal Domain Adaptation) datasets
    """

    def __init__(self, domain_type="target", dataset="Office", data_dir="", data_list=[],
                 shared_class_num=0, source_private_class_num=0, target_private_class_num=0,
                 unida_setting="OSDA", preload_flg=False, eval=True):
    
        super().__init__()
        
        # dataset vars
        self.domain_type = domain_type # source or target
        self.dataset = dataset # "Office", "Office-Home", "VisDA" or "DomainNet"
        self.preload_flg = preload_flg # whether to pre-load images into memory
        
        # number of evaluated classes
        self.shared_class_num = shared_class_num
        self.source_private_class_num = source_private_class_num
        self.target_private_class_num = target_private_class_num 
        
        # labels of source classes (shared + private)
        self.shared_classes = [label for label in range(shared_class_num)] # from 0 to shared_class_num - 1
        self.source_private_classes = [label + shared_class_num for label in range(source_private_class_num)]
        
        # obtaining labels of target classes
        if dataset == "Office" and unida_setting == "OSDA":
            # OSDA in Office omits the middle classes (from 10 to 19) for evaluation
            # target priv classes range from 20 to 30 (shared_class_num is 10 and source_private_class_num = 0)
            self.target_private_classes = [i + shared_class_num + source_private_class_num + 10 for i in range(target_private_class_num)]
        else:
            self.target_private_classes = [i + shared_class_num + source_private_class_num for i in range(target_private_class_num)]
        
        # consolidating class labels
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir # defining data directory
        
        # obtaining items of current domain from main data_list
        self.data_list = [item.strip().split() for item in data_list]
        if self.domain_type == "source": 
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
        else: 
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        self.pre_loading()
        self.test_transform = preprocess_images_eval()
        
    def pre_loading(self):
        """
        Pre-load images if preload_flg is set to True
        NOTE: Office is small enough to pre-load all images into memory.
        """

        if "Office" in self.dataset and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")

            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        """
        Load an image and its label from the dataset
        """

        img_path, img_label = self.data_list[img_idx]
        if "Office" in self.dataset and self.preload_flg:
            # using pre-loaded images for Office
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")   
        return img, img_label
    
    def __len__(self):
        """
        Return the total number of images in the dataset
        """
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        """
        Get an image and its label by index
        """

        img, img_label = self.load_img(img_idx) # remove full img path after debugging
        
        if self.domain_type == "source":
            img_label = int(img_label)
        else:
            # in the target domain, unknown classes have a label of len(self.source_classes)
            img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)
        
        img_test = self.test_transform(img)

        return img_test, img_label, img_idx


if __name__ == "__main__":
    print("Module with UniDA dataset classes")