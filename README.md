# Vestibular Schwannoma Segmentation

## Overview

This repository provides tools and scripts for the segmentation of vestibular schwannomas from medical images using state-of-the-art deep learning models. The aim is to automate and enhance the accuracy of vestibular schwannoma detection and segmentation in MRI scans.

We have customized and extended the original **MedSAM** and **UNETR** repositories to suit the needs of this project. Below are references to the original works:

- **MedSAM**: [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)  
  _Segment Anything in Medical Images_  
  **Citation**:  

@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}

- **UNETR**: [UNETR GitHub](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)  
_UNETR: Transformers for 3D Medical Image Segmentation_  
**Citation**:  
@inproceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={574--584},
  year={2022}
}

## Features

- Automated segmentation of vestibular schwannomas using **UNETR** and **MedSAM** models.
- Customized code for training and inference tailored to vestibular schwannoma segmentation tasks.
- Tools for preprocessing, visualization, and postprocessing.
- Evaluation metrics including **Dice**, **Hausdorff Distance**, **95th Percentile Hausdorff Distance**, **Slice-to-Slice Dice (S2S)**, and **Relative Volume Error (RVE)**.

```plaintext
VestibularSchwannomaSegmentation/
├── UNETR/                       # UNETR-based segmentation models
├── MedSAM/                      # MedSAM segmentation tools
├── calculate_result_metrics.py  # Script for calculating segmentation metrics: DICE, Hausdorff, Hausdorff95, S2S, RVE scores
└── README.md                    # Project description
```


## Usage

### **MedSAM**
1. Prepare your data:
   - Place your data under `./MedSAM/data/npy/MRI_VestSch` in the format specified by the original repository:
     - Use `imgs` and `gts` subdirectories.
     - Each slice must be saved as `.npy` files with names like `image_name-000.npy`, `image_name-001.npy` to indicate slice numbers.
2. Download the SAM checkpoint:
   - [Download SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
   - Place it at `work_dir/SAM/sam_vit_b_01ec64.pth`.
3. Training:
   - Run the training script: `./MedSAM/1_MedSAM_training.py`.
4. Inference:
   - Run the inference notebook: `./MedSAM/2_MedSAM_inference.ipynb`.
   - This notebook will also execute `calculate_result_metrics.py` to evaluate the results.

### **UNETR**
1. Install requirements:
   - Requirements are listed in `./UNETR/requirements.txt`.
2. Prepare your data:
   - Place your data under `./UNETR/BTCV/dataset` as follows:
     - Use `imagesTr`, `labelsTr`, `imagesTs`, and `labelsTs` subdirectories for training and testing NIfTI files.
     - Create a dataset JSON file as specified in the original repository (details are provided in the training notebook).
3. Training:
   - Run the training notebook: `./UNETR/1_UNETR_training.ipynb`.
4. Inference:
   - Run the inference notebook: `./UNETR/2_UNETR_inference.ipynb`.
   - This notebook will also execute `calculate_result_metrics.py` to evaluate the results.


## Evaluation Metrics

The script `calculate_result_metrics.py` computes the following metrics for evaluating segmentation performance:

- **Dice Score**: Measures the overlap between predicted and ground truth masks.
- **Hausdorff Distance (HD)**: Maximum distance between the boundaries of predicted and ground truth masks.
- **95th Percentile Hausdorff Distance (HD95)**: 95th percentile of distances for robust evaluation.
- **Slice-to-Slice Dice (S2S)**: Per-slice Dice score for 3D images.
- **Relative Volume Error (RVE)**: Difference in predicted and ground truth volumes as a percentage.

## Acknowledgments

This repository builds upon the works of **MedSAM** and **UNETR**. We acknowledge the contributions of the authors and provide due credit for their foundational efforts.

- MedSAM: [GitHub](https://github.com/bowang-lab/MedSAM)
- UNETR: [GitHub](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)
