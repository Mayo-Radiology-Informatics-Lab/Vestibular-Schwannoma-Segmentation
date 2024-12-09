# -*- coding: utf-8 -*-
"""
Train the image encoder and mask decoder.
Freeze the prompt image encoder during training.

This script is designed to train a medical segmentation model (MedSAM) using 
a ViT-based Segment Anything Model (SAM) architecture. It supports features such as:
- Mixed precision training with AMP
- Time-limited training for controlled resource usage
- Pretrained model loading
- Custom dataset handling with .npy format
- Weights & Biases integration for experiment tracking

Reference:
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}

This code has been prepared based on the following original study:
https://github.com/bowang-lab/MedSAM
"""

# %% Setup Environment
import os
import glob
import shutil
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from skimage import transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry

# Set random seeds and environment variables
torch.manual_seed(666)
torch.cuda.manual_seed_all(666)
np.random.seed(666)
random.seed(666)

torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


# %% Argument Parser
"""
Arguments are parsed to allow flexible configuration of the training process.
- Paths, model types, and hyperparameters can be adjusted via command-line arguments.
"""
parser = argparse.ArgumentParser(description="MedSAM Training Script")
parser.add_argument(
    "-i", "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="Path to training .npy files (expects 'gts' and 'imgs' subfolders).",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B", help="Task name.")
parser.add_argument("-model_type", type=str, default="vit_b", help="Model type.")
parser.add_argument(
    "-checkpoint", type=str,
    default="work_dir/SAM/sam_vit_b_01ec64.pth",
    help="Path to model checkpoint."
)
parser.add_argument("--load_pretrain", type=bool, default=True, help="Load pretrained model.")
parser.add_argument("-pretrain_model_path", type=str, default="", help="Path to pretrained model.")
parser.add_argument("-work_dir", type=str, default="./work_dir", help="Working directory.")
parser.add_argument("-num_epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("-batch_size", type=int, default=2, help="Batch size for training.")
parser.add_argument("-num_workers", type=int, default=0, help="Number of data loader workers.")
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)."
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="Learning rate."
)
parser.add_argument(
    "-use_wandb", type=bool, default=True, help="Use Weights & Biases for monitoring."
)
parser.add_argument("-use_amp", action="store_true", default=True, help="Use AMP for mixed precision.")
parser.add_argument(
    "--resume", type=str, default="", help="Resume training from a checkpoint."
)
parser.add_argument(
    "-max_time", type=int, default=0,
    help="Maximum training duration in minutes. Set to 0 for no limit."
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0' or 'cpu').")

args = parser.parse_args()

# %% GPU Device Selection
"""
Automatically detects available GPUs and assigns the first available device for training.
If no GPU is available, training defaults to CPU. This ensures compatibility with various
hardware setups.
"""
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        gpu_properties = torch.cuda.get_device_properties(i)
        if torch.cuda.memory_reserved(i) == 0:  # Check if GPU memory is available
            device = torch.device(f"cuda:{i}")
            print(f"Training on GPU: {gpu_properties.name}")
            break
    else:
        print("No free GPU found, training on CPU.")
        device = torch.device("cpu")
else:
    print("No GPU available, training on CPU.")
    device = torch.device("cpu")


# %% Working Directory Setup
"""
Creates a unique working directory for this training session, using the current timestamp.
This ensures that training artifacts (models, logs, etc.) are properly organized.
"""
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = os.path.join(args.work_dir, f"{args.task_name}-{run_id}")
os.makedirs(model_save_path, exist_ok=True)

# %% Weights & Biases Setup (Optional)
"""
Integrates Weights & Biases for experiment tracking. Users can monitor loss curves,
learning rates, and other metrics in real time. To disable, set `--use_wandb False`.
"""
if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% Utility Functions
"""
Helper functions for visualization (masks, bounding boxes) and time checking.
- `show_mask`: Overlays a segmentation mask on an image.
- `show_box`: Draws a bounding box around a region of interest.
- `time_exceeded`: Checks if the training duration has exceeded the specified time limit.
"""
def show_mask(mask, ax, random_color=False):
    """Overlay a mask on a matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """Draw a bounding box on a matplotlib axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

def time_exceeded(start_time):
    return (datetime.now() - start_time).total_seconds() > args.max_time * 60

# %% Dataset Class
"""
Dataset class to handle .npy image and ground truth data.

This class is designed for .npy files with the following folder structure:
- 'imgs': Contains the input images.
- 'gts': Contains the ground truth masks corresponding to the images.

Features:
- Automatically matches ground truth files with corresponding images.
- Generates bounding boxes around non-zero regions in the ground truth masks.
- Allows random perturbation (shifting) of bounding box coordinates to add variability.
"""
class NpyDataset(Dataset):
    """
    Dataset class to handle .npy image and ground truth data.

    Args:
    - data_root (str): Path to dataset directory with 'gts' and 'imgs' subfolders.
    - bbox_shift (int): Random shift applied to bounding box coordinates.
    """
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift

        print(f"Number of images: {len(self.gt_path_files)}")
        print(f"Ground truth path: {self.gt_path}")
        print(f"Image path: {self.img_path}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(os.path.join(self.img_path, img_name), allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # Convert to (C, H, W)

        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, \
            "Image values should be normalized to [0, 1]"

        gt = np.load(self.gt_path_files[index], allow_pickle=True)
        label_ids = np.unique(gt)[1:]  # Skip background (0)
        gt2D = np.uint8(gt == random.choice(label_ids.tolist()))  # Select one label

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Apply random perturbation to bounding box
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% Sanity Check for Dataset
"""
Performs a sanity check on the dataset by visualizing a few examples.
This helps ensure that the dataset is correctly loaded and processed.
- Visualizes images, overlaid ground truth masks, and bounding boxes.
- Saves the visualization to 'data_sanitycheck.png'.
"""
tr_dataset = NpyDataset(args.tr_npy_path)
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)

# Perform sanity check
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    # Batch shapes
    print(f"Image Shape: {image.shape}, GT Shape: {gt.shape}, BBoxes Shape: {bboxes.shape}")
    
    # Create figure with 2 subplots
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    
    # Randomly select an example for the first subplot
    idx = random.randint(0, len(image) - 1)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())  # Visualize image
    show_mask(gt[idx].cpu().numpy(), axs[0])  # Overlay mask
    show_box(bboxes[idx].numpy(), axs[0])  # Draw bounding box
    axs[0].axis("off")
    axs[0].set_title(names_temp[idx])  # Set title with filename
    
    # Randomly select another example for the second subplot
    idx = random.randint(0, len(image) - 1)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())  # Visualize image
    show_mask(gt[idx].cpu().numpy(), axs[1])  # Overlay mask
    show_box(bboxes[idx].numpy(), axs[1])  # Draw bounding box
    axs[1].axis("off")
    axs[1].set_title(names_temp[idx])  # Set title with filename
    
    # Adjust layout and save the figure
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Sanity check visualization saved at ./data_sanitycheck.png")
    break  # Process only the first batch



# %% Model Definition
"""
Defines the MedSAM model, which integrates an image encoder, mask decoder,
and a frozen prompt encoder. The prompt encoder is frozen to reduce
computational cost and focus training on the image and mask decoders.
"""
class MedSAM(nn.Module):
    """
    Medical SAM (Segment Anything Model) with frozen prompt encoder.

    Args:
    - image_encoder (nn.Module): Encoder for image embeddings.
    - mask_decoder (nn.Module): Decoder for generating masks.
    - prompt_encoder (nn.Module): Encoder for prompts (frozen).
    """
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # Freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            box_torch = box_torch[:, None, :] if len(box_torch.shape) == 2 else box_torch

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


# %% Training Loop
"""
Main training loop for the MedSAM model.

Features:
- Logs training progress, including loss metrics.
- Saves model checkpoints after every epoch (latest and best models).
- Supports time-limited training for better resource management.
- Plots and saves loss curves for monitoring.
"""
def train_model():
    """
    Train the MedSAM model.
    """
    os.makedirs(model_save_path, exist_ok=True)
    if '__file__' in globals():
        shutil.copyfile(__file__, os.path.join(model_save_path, run_id + "_" + os.path.basename(__file__)))
    else:
        print("Warning: __file__ is not defined. Skipping script file backup.")

    # Initialize SAM and MedSAM models
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    # Optimizer and Loss Functions
    optimizer = torch.optim.AdamW(
        list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Training Loop
    start_time = datetime.now()
    best_loss = float("inf")
    losses = []

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(tr_dataloader)):
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device)
            boxes_np = boxes.detach().cpu().numpy()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        # Log and Save
        epoch_loss /= len(tr_dataloader)
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})

        print(f"Epoch {epoch}, Loss: {epoch_loss}")
        torch.save(medsam_model.state_dict(), os.path.join(model_save_path, "medsam_model_latest.pth"))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(medsam_model.state_dict(), os.path.join(model_save_path, "medsam_model_best.pth"))

        # Time Limit Check
        if (datetime.now() - start_time).total_seconds() > args.max_time * 60:
            print(f"Training stopped after reaching the time limit of {args.max_time} minutes.")
            break

    # Plot Losses
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_save_path, args.task_name + "_train_loss.png"))
    plt.close()

if __name__ == "__main__":
    parser.set_defaults(
        tr_npy_path="/research/projects/Sahika/MedSAM/data/npy/MRI_VestSch",
        task_name="MedSAM-ViT-B",
        model_type="vit_b",
        pretrain_model_path="/research/projects/Sahika/MedSAM/work_dir/MedSAM/sam_vit_b_01ec64.pth",
        work_dir="/research/projects/Sahika/MedSAM/work_dir",
        batch_size=1,
        num_epochs=200,
        weight_decay=0.01,
        max_time=7200,
    )
    args = parser.parse_args()

    train_model()
