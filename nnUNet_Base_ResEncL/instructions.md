# 🧠 nnU-Net Model Reproduction Guide

This document describes how to reproduce the **nnU-Net**–based models used in our study, including the **nnUNet Base** and **nnUNet ResEncL** configurations.

---

## 1.Environment Setup

Create and activate the conda environment using the provided YAML file (`nnunet_env.yaml`):

```bash
conda env create -f nnunet_env.yaml
conda activate nnunet
```

---

## 2.Clone the Official nnU-Net Repository

Clone the official nnU-Net v2 repository maintained by MIC-DKFZ and install it into the environment:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

> ℹ️ **Reference:** Official nnU-Net v2 repository by MIC-DKFZ  
> https://github.com/MIC-DKFZ/nnUNet

This repository provides the official implementation, documentation, and command-line tools for data preprocessing, training, and inference.

---

## 3.Dataset Preparation and Preprocessing

Organize your dataset following the nnU-Net folder structure, as mentioned in the original repository:

```
nnUNet_raw/
├── Dataset<DATASET_ID_WITH_3_DIGITS>_<DATASET_NAME>/
│   ├── imagesTr/
│   ├── imagesTs/
│   ├── labelsTr/
│   └── labelsTs/
```

Then run preprocessing (adjust dataset ID as needed):

```bash
nnUNetv2_plan_and_preprocess -d 300 --verify_dataset_integrity
```

---

## 4.Model Training

### 🔹 nnUNet Base

```bash
nnUNetv2_train -d <DATASET_ID>  3d_fullres all -device cuda -tr nnUNetTrainer --npz --c
```

### 🔹 nnUNet ResEncL

```bash
nnUNetv2_train -d <DATASET_ID>  3d_fullres all -device cuda -tr nnUNetTrainer -p nnUNetResEncUNetLPlans --npz --c
```

Training automatically applies nnU-Net’s self-configuration for patch size, normalization, and network architecture.  
Each model was trained on a dedicated GPU (A100 80GB) for reproducibility and isolation across frameworks.

---

## 5.Model Inference (Prediction)

### 🔹 nnUNet Base Inference

```bash
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/output_folder -d <DATASET_ID> -c 3d_fullres -f all
```

### 🔹 nnUNet ResEncL Inference

```bash
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/output_folder -d <DATASET_ID> -c 3d_fullres -f all
```

---

## 6.Additional Details

- Detailed documentation for preprocessing, training, and prediction is available in the official nnU-Net repository.
- For technical details, refer to:

> **Isensee F, et al.**  
> *nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation.*  
> *Nature Methods*, 2021.  
> https://github.com/MIC-DKFZ/nnUNet

---

**Note:**  
For changing epoch numbers, you can define custom trainer here: .../miniconda3/envs/nnunet/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py

Change this line: self.num_epochs = 8000 or self.num_epochs = 3500
