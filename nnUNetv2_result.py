import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import json
from tqdm.cli import tqdm
import argparse
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
import torch

# Resampling function to match dimensions
def resample_to_reference(image, reference_image):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    return resample.Execute(image)

# Dice score calculation function
def calculate_dice_score(pred, gt):
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    elif pred_sum == 0 or gt_sum == 0:
        return 0.0
    intersection = np.logical_and(pred, gt)
    dice = 2. * intersection.sum() / (pred.sum() + gt.sum())
    return dice

# Dice score calculation function using MONAI DiceMetric
def calculate_dice_score_monai(pred, gt):
    # Initialize DiceMetric with MONAI
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Convert ground truth and prediction to PyTorch tensors
    gt_tensor = torch.tensor(gt, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    pred_tensor = torch.tensor(pred, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    
    # Ensure the prediction is binarized
    pred_tensor = AsDiscrete(threshold=0.5)(pred_tensor)

    # Compute the Dice score
    dice_metric(y_pred=pred_tensor, y=gt_tensor)
    dice_score = dice_metric.aggregate().item()

    # Reset the metric for the next image
    dice_metric.reset()

    return dice_score

# Hausdorff distance calculation function
def calculate_hausdorff_distance(pred, gt):
    if np.count_nonzero(pred) == 0:
        print("Predicted mask is empty, skipping Hausdorff calculation.")
        return None
    if np.count_nonzero(gt) == 0:
        print("Ground truth mask is empty, skipping Hausdorff calculation.")
        return None
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
    gt_sitk = sitk.GetImageFromArray(gt.astype(np.uint8))
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(pred_sitk, gt_sitk)
    return hausdorff_distance_filter.GetHausdorffDistance()

def calculate_hausdorff95_distance(pred, gt):
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
    gt_sitk = sitk.GetImageFromArray(gt.astype(np.uint8))
    
    # Surface distance computation
    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_sitk, squaredDistance=False, useImageSpacing=True))
    surface_distance = sitk.LabelContour(gt_sitk, fullyConnected=False)
    
    surface_mask = sitk.GetArrayFromImage(surface_distance).astype(bool)
    distances = sitk.GetArrayViewFromImage(distance_map)[surface_mask]
    
    if distances.size == 0:
        raise ValueError("Empty distance array - no surface points found.")
    
    hd95 = np.percentile(distances, 95)
    
    return hd95

# Slice-to-Slice Dice score calculation function
def calculate_s2s_dice_score(pred, gt):
    slice_dice_scores = []
    
    # Ensure pred and gt have the same number of slices
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch in S2S Dice calculation: pred.shape={pred.shape}, gt.shape={gt.shape}")
    
    # Iterate through slices and calculate Dice for each slice
    for z in range(pred.shape[0]):  # Iterate over the z-axis (slice index)
        pred_slice = pred[z, :, :]
        gt_slice = gt[z, :, :]
        
        # Calculate the Dice score for the current slice
        dice = calculate_dice_score(pred_slice, gt_slice)
        slice_dice_scores.append(dice)
    
    return np.mean(slice_dice_scores)

# RVE (Relative Volume Error) calculation function with overflow protection
def calculate_rve(pred, gt):
    pred_volume = np.sum(pred)
    gt_volume = np.sum(gt)

    # Avoid overflow by setting a maximum threshold for volume values
    max_volume = 1e10  # Example threshold to limit extremely large values
    pred_volume = np.clip(pred_volume, 0, max_volume)
    gt_volume = np.clip(gt_volume, 0, max_volume)

    if gt_volume == 0:
        if pred_volume == 0:
            return 0.0  # Both are zero, so no error
        else:
            return float('inf')  # Infinite error if ground truth is zero and prediction is not

    try:
        rve = abs(pred_volume - gt_volume) / gt_volume
    except OverflowError:
        # Handle overflow errors by returning a large value or a flag
        rve = float('inf')
    
    return rve

# Helper function to parse thresholds from a string to a list
def parse_thresholds(threshold_str):
    return list(map(int, threshold_str.strip('[]').split(',')))

# Main processing function
def main(gt_folder, pred_folder, csv_file, thresholds):
    df = pd.read_csv(csv_file)
    
    # Dynamically create metrics for each threshold
    metrics = {
    'all': {'dice': [], 'hausdorff': [], 'hausdorff95': [], 'S2S': [], 'RVE': []},
    }
    metrics[f'<{thresholds[0]}'] = {'dice': [], 'hausdorff': [], 'hausdorff95': [], 'S2S': [], 'RVE': []}
    for i in range(len(thresholds) - 1):
        metrics[f'{thresholds[i]}-{thresholds[i+1]}'] = {'dice': [], 'hausdorff': [], 'hausdorff95': [], 'S2S': [], 'RVE': []}
    metrics[f'>{thresholds[-1]}'] = {'dice': [], 'hausdorff': [], 'hausdorff95': [], 'S2S': [], 'RVE': []}
   
    missing_files = []
    image_metrics = []
    count = 0
    # Main loop to process each image
    for index, row in tqdm(df.iterrows()):
        image_name = row['file_name']
        tumor_size = row.get('tumor_volume') or row.get('mask_volume')

        ground_truth_path = os.path.join(gt_folder, image_name)
        predicted_mask_path = ground_truth_path.replace(gt_folder, pred_folder)

        if os.path.exists(ground_truth_path) and os.path.exists(predicted_mask_path):
            try:
                ground_truth = sitk.ReadImage(ground_truth_path, sitk.sitkUInt8)
                predicted_mask = sitk.ReadImage(predicted_mask_path, sitk.sitkUInt8)

                if ground_truth.GetSize() != predicted_mask.GetSize():
                    print('Mask sizes not matched, resampling...')
                    predicted_mask = resample_to_reference(predicted_mask, ground_truth)

                ground_truth_array = sitk.GetArrayFromImage(ground_truth)
                predicted_mask_array = sitk.GetArrayFromImage(predicted_mask)

                dice = calculate_dice_score_monai(predicted_mask_array, ground_truth_array)
                s2s = calculate_s2s_dice_score(predicted_mask_array, ground_truth_array)
                rve = calculate_rve(predicted_mask_array, ground_truth_array)

                if np.count_nonzero(ground_truth_array) > 0 and np.count_nonzero(predicted_mask_array) > 0:
                    hausdorff = calculate_hausdorff_distance(predicted_mask_array, ground_truth_array)
                    hausdorff95 = calculate_hausdorff95_distance(predicted_mask_array, ground_truth_array)
                else:
                    print(f"Skipping Hausdorff calculation for {image_name} due to empty mask. GT tumor size: {tumor_size}")
                    hausdorff, hausdorff95 = None, None

                # Store metrics in the appropriate category
                if tumor_size < thresholds[0]:
                    metrics[f'<{thresholds[0]}']['dice'].append(dice)
                    metrics[f'<{thresholds[0]}']['S2S'].append(s2s)
                    metrics[f'<{thresholds[0]}']['RVE'].append(rve)
                    if hausdorff is not None:
                        metrics[f'<{thresholds[0]}']['hausdorff'].append(hausdorff)
                        metrics[f'<{thresholds[0]}']['hausdorff95'].append(hausdorff95)
                    else:
                        metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff'].append(None)
                        metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff95'].append(None)
                else:
                    for i in range(len(thresholds) - 1):
                        if thresholds[i] <= tumor_size < thresholds[i + 1]:
                            metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['dice'].append(dice)
                            metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['S2S'].append(s2s)
                            metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['RVE'].append(rve)
                            if hausdorff is not None:
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff'].append(hausdorff)
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff95'].append(hausdorff95)
                            else:
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff'].append(None)
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff95'].append(None)
                            break
                    else:
                        if tumor_size >= thresholds[-1]:
                            metrics[f'>{thresholds[-1]}']['dice'].append(dice)
                            metrics[f'>{thresholds[-1]}']['S2S'].append(s2s)
                            metrics[f'>{thresholds[-1]}']['RVE'].append(rve)
                            if hausdorff is not None:
                                metrics[f'>{thresholds[-1]}']['hausdorff'].append(hausdorff)
                                metrics[f'>{thresholds[-1]}']['hausdorff95'].append(hausdorff95)
                            else:
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff'].append(None)
                                metrics[f'{thresholds[i]}-{thresholds[i + 1]}']['hausdorff95'].append(None)

                metrics['all']['dice'].append(dice)
                metrics['all']['S2S'].append(s2s)
                metrics['all']['RVE'].append(rve)
                if hausdorff is not None:
                    metrics['all']['hausdorff'].append(hausdorff)
                    metrics['all']['hausdorff95'].append(hausdorff95)

                # Store the metrics for each image
                image_metrics.append((image_name, dice, s2s, rve, hausdorff, hausdorff95))
                # Store the metrics for each image
                #image_metrics.append({
                #    "image_name": image_name,
                #    "tumor_size": tumor_size,
                #    "dice": dice,
                #    "s2s": s2s,
                #    "rve": rve,
                #    "hausdorff": hausdorff,
                #    "hausdorff95": hausdorff95
                #})
                # Increment the count of processed images
                count += 1

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
        else:
            missing_files.append(image_name)

    # Output missing files if any
    if missing_files:
        print(f"Missing files: {missing_files}")

    print(f"Processed {count} images.")

    # Save results to the pred_folder
    output_file = os.path.join(pred_folder, "results.json")

    averaged_metrics = {}
    for group, group_metrics in metrics.items():
        averaged_metrics[group] = {}
        for metric, values in group_metrics.items():
            if values:
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    averaged_metrics[group][metric] = np.mean(valid_values)
                else:
                    averaged_metrics[group][metric] = None  # Or set to 0, depending on your preference
            else:
                averaged_metrics[group][metric] = None

    summary = {}
    for group, values in metrics.items():
        if values['dice']:  # Only calculate the mean if there are values
            mean_dice = round(np.mean(values['dice']), 4)
        else:
            mean_dice = 0.0  # Set to 0 if no values exist

        summary[f"{group}_mean_dice"] = mean_dice
        summary[f"{group}_mean_S2S"] = round(np.mean([v for v in values['S2S'] if v is not None]), 4) if values['S2S'] else 0.0
        summary[f"{group}_mean_RVE"] = round(np.mean([v for v in values['RVE'] if v is not None]), 4) if values['RVE'] else 0.0

        if group == 'all' and values['hausdorff']:  # Special case for all
            mean_hausdorff = round(np.mean([v for v in values['hausdorff'] if v is not None]), 4)
            mean_hausdorff95 = round(np.mean([v for v in values['hausdorff95'] if v is not None]), 4)
            summary["mean_hausdorff"] = round(mean_hausdorff, 4)
            summary["mean_hausdorff95"] = round(mean_hausdorff95, 4)



    # Output the results
    output_data = {
        "Averaged Metrics": averaged_metrics,
        "Summary": summary,
        "Image-specific Metrics": image_metrics
    }

    # Replace NaN values with None and ensure float32 values are converted to float
    output_data = json.loads(json.dumps(output_data, default=lambda o: None if np.isnan(o) else float(o) if isinstance(o, (np.float32, torch.Tensor)) else o))

    #with open(output_file, 'w') as f:
        #json.dump(output_data, f, indent=4)

    # Print the summary with labels
    print(f"Results saved to {output_file}")
    print('Summary:')

    # Print the overall dice score first
    if 'all_mean_dice' in summary:
        print(f"All Mean Dice: {summary['all_mean_dice']}")

    # Print dice scores for each size category in the desired order
    if f'<{thresholds[0]}_mean_dice' in summary:
        print(f"<{thresholds[0]} Dice: {summary[f'<{thresholds[0]}_mean_dice']}")
    if f'{thresholds[0]}-{thresholds[1]}_mean_dice' in summary:
        print(f"{thresholds[0]}-{thresholds[1]} Dice: {summary[f'{thresholds[0]}-{thresholds[1]}_mean_dice']}")
    if f'>{thresholds[-1]}_mean_dice' in summary:
        print(f">{thresholds[-1]} Dice: {summary[f'>{thresholds[-1]}_mean_dice']}")

    # Finally, print the Hausdorff metrics
    if "mean_hausdorff" in summary:
        print(f"Mean Hausdorff: {summary['mean_hausdorff']}")
    if "mean_hausdorff95" in summary:
        print(f"Mean Hausdorff95: {summary['mean_hausdorff95']}")
    if "all_mean_S2S" in summary:
        print(f"All Mean S2S: {summary['all_mean_S2S']}")
    if "all_mean_RVE" in summary:
        print(f"All Mean RVE: {summary['all_mean_RVE']}")

    # Print the summary without labels (just the values)
    print("\nSummary without labels:")
    if 'all_mean_dice' in summary:
        print(f"{summary['all_mean_dice']}")

    # Print dice scores for each size category without labels
    if f'<{thresholds[0]}_mean_dice' in summary:
        print(f"{summary[f'<{thresholds[0]}_mean_dice']}")
    if f'{thresholds[0]}-{thresholds[1]}_mean_dice' in summary:
        print(f"{summary[f'{thresholds[0]}-{thresholds[1]}_mean_dice']}")
    if f'>{thresholds[-1]}_mean_dice' in summary:
        print(f"{summary[f'>{thresholds[-1]}_mean_dice']}")

    # Print the Hausdorff metrics without labels
    if "mean_hausdorff" in summary:
        print(f"{summary['mean_hausdorff']}")
    if "mean_hausdorff95" in summary:
        print(f"{summary['mean_hausdorff95']}")
    if "all_mean_S2S" in summary:
        print(f"{summary['all_mean_S2S']}")
    if "all_mean_RVE" in summary:
        print(f"{summary['all_mean_RVE']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and calculate Dice, Hausdorff, and Hausdorff95 distances.")
    parser.add_argument('-gt', '--ground_truth_folder', type=str, required=True, help="Folder containing ground truth masks.")
    parser.add_argument('-pred', '--pred_folder', type=str, required=True, help="Folder containing predicted segmentation masks.")
    parser.add_argument('-csv', '--csv_file', type=str, required=True, help="CSV file with image names and tumor volumes.")
    parser.add_argument('-th', '--thresholds', type=parse_thresholds, default=[200, 400], help="List of thresholds for categorizing tumor size. Example: [200,400]")

    args = parser.parse_args()

    main(args.ground_truth_folder, args.pred_folder, args.csv_file, args.thresholds)