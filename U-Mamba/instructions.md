# 🧠 U-Mamba Model Reproduction Guide

This document describes how to reproduce the **nnU-Net**–based models used in our study, including the **nnUNet Base** and **nnUNet ResEncL** configurations.

---

## 1.Environment Setup

Create and activate the conda environment using the provided YAML file (`umamba_env.yaml`):

```bash
conda env create -f umamba_env.yaml
conda activate umamba
```

---

## 2.Clone the Official nnU-Net Repository

Clone the official nnU-Net v2 repository maintained by MIC-DKFZ and install it into the environment:

```bash
git clone https://github.com/bowang-lab/U-Mamba
cd U-Mamba
pip install -e .
```

> ℹ️ **Reference:** Official U-Mamba repository by bowang-lab  
> https://github.com/bowang-lab/U-Mamba

This repository provides the official implementation, documentation, and command-line tools for data preprocessing, training, and inference.

---

## 3.Dataset Preparation and Preprocessing

Organize your dataset following the U-Mamba folder structure:

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
nnUNetv2_plan_and_preprocess -d <DATASET_ID>  --verify_dataset_integrity
```

---

## 4.Model Training

```bash
nnUNetv2_train -d <DATASET_ID>  3d_fullres all -device cuda -tr nnUNetTrainerUMambaBot --npz --c
```

Training automatically applies nnU-Net’s self-configuration for patch size, normalization, and network architecture.  

---

## 5.Model Inference (Prediction)

```bash
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/output_folder -d <DATASET_ID> -c 3d_fullres -f all
```

---

## 6.Additional Details

- Detailed documentation for preprocessing, training, and prediction is available in the official nnU-Net repository.
- For technical details, refer to:

> **Isensee F, et al.**  
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}

---

**Note:**  
For changing epoch numbers, you can define custom trainer here: .../miniconda3/envs/umamba/lib/python3.10/site-packages/U-Mamba/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py
