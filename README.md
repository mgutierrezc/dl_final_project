# Codebase for UniDA + TTA project

# GitHub Repo

Click [here](https://github.com/mgutierrezc/dl_final_project) to access the public repository of this project

# Setup

1. Load anaconda on your HPC terminal using `module load miniforge` and run `conda env create -f environment.yaml`. If the installation run into any issues with a specific issues with a package, comment it out from the `environment.yaml` and rerun `conda env create -f environment.yaml`
2. Activate the environment

# Download Datasets

- Office-31 can be found in [Kaggle](https://www.kaggle.com/datasets/xixuhu/office31)
- The VisDA we use corresponds to the classification track of 2017 and can be found [here](https://ai.bu.edu/visda-2017/#classification)

The download is straightforward but might take from some hours to a day depending on the internet connection of the HPC.

# Training

1. Clone the repo from LEAD on your HPC TERMINAL: `git clone git@github.com:ispc-lab/LEAD.git`
2. Move into the directory of LEAD
3. Run the following commands

## Open-Partial Domain Adaptation
```bash
# Source Model Preparing
bash ./scripts/train_source_OPDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OPDA.sh
```

## Open-Set Domain Adaptation
```bash
# Source Model Preparing
bash ./scripts/train_source_OSDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OSDA.sh
```

## Partial Domain Adaptation
```bash
# Source Model Preparing
bash ./scripts/train_source_PDA.sh
# Target Model Adaptation
bash ./scripts/train_target_PDA.sh
```

This will also output the results for the no corruption scenario

## Generate Corruptions

```bash
# Office
bash ./scripts/generate_corruptions_office.sh
# VisDA
bash ./scripts/generate_corruptions_visda.sh
```

## Corruption runs

To obtain our corruption runs, we'll need to run our trained LEAD model for every corrupted version of Office and VisDA

```bash
# Office
bash ./scripts/lead_corruption_runs_office.sh
# VisDA
bash ./scripts/lead_corruption_runs_visda.sh
```

## Grid Search

The TTA methods neeed a grid search implemented in the Gaussian Noise corruption of every dataset

```bash
# Office
bash ./scripts/lead_corruption_runs_office_gs_tent.sh
bash ./scripts/lead_corruption_runs_office_gs_eata.sh
bash ./scripts/lead_corruption_runs_office_gs_sotta.sh
bash ./scripts/lead_corruption_runs_office_gs_stamp.sh
bash ./scripts/lead_corruption_runs_office_gs_gmm.sh
# VisDA
bash ./scripts/lead_corruption_runs_visda_gs_tent.sh
bash ./scripts/lead_corruption_runs_visda_gs_eata.sh
bash ./scripts/lead_corruption_runs_visda_gs_sotta.sh
bash ./scripts/lead_corruption_runs_visda_gs_stamp.sh
bash ./scripts/lead_corruption_runs_visda_gs_gmm.sh
```

## Corruption + TTA Runs

Finally, we'll obtain our Corruption + TTA runs results from all corruptions by running the following lines.

```bash
# Office
bash ./scripts/lead_corruption_runs_office_tent.sh
bash ./scripts/lead_corruption_runs_office_eata.sh
bash ./scripts/lead_corruption_runs_office_sotta.sh
bash ./scripts/lead_corruption_runs_office_stamp.sh
bash ./scripts/lead_corruption_runs_office_gmm.sh
# VisDA
bash ./scripts/lead_corruption_runs_visda_tent.sh
bash ./scripts/lead_corruption_runs_visda_eata.sh
bash ./scripts/lead_corruption_runs_visda_sotta.sh
bash ./scripts/lead_corruption_runs_visda_stamp.sh
bash ./scripts/lead_corruption_runs_visda_gmm.sh
```

# Code Citations

They have been included in every script according to the resource used, but will also be summarized below

For Pytorch:
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. arXiv preprint arXiv:1912.01703. https://arxiv.org/abs/1912.01703

The classes for the TTA come from https://github.com/yuyongcan/STAMP/tree/master
The only TTA class that has a separate source is GMM and comes from https://github.com/pascalschlachter/GMM/blob/main/adaptation.py
The Unida models are based on the code from https://github.com/ispc-lab/LEAD and https://github.com/ispc-lab/GLC
The classes and functions for Corruptions come from https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
The functions for GMM related tasks come from https://github.com/pascalschlachter/GMM/blob/main/adaptation.py