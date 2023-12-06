import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def get_images_with_suffix(directory, suffix):
    pattern = os.path.join(directory, f'**/*{suffix}')
    image_files = sorted(glob.glob(pattern, recursive=True))
    return image_files

class GtDataset(Dataset):
    def __init__(self, file_list, max_len=-1):
        self.file_list = file_list
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list) if self.max_len == -1 else min(len(self.file_list), self.max_len)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert('L')
        img_array = torch.tensor(np.array(img), dtype=torch.uint8)

        return img_array

class PredDataset(Dataset):
    def __init__(self, file_list, palette, max_len=-1):
        self.file_list = file_list
        self.palette = palette
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list) if self.max_len == -1 else min(len(self.file_list), self.max_len)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = Image.open(file_path).convert('RGB')
        img_array = torch.tensor(np.array(img), dtype=torch.uint8)

        # Map RGB values to class indices based on the palette
        label_array = torch.zeros_like(img_array[:, :, 0], dtype=torch.uint8)
        for idx, color in enumerate(self.palette):
            mask = torch.all(img_array == torch.tensor(color), dim=-1)
            label_array[mask] = idx

        return label_array

def calculate_accuracy(gt_labels, pred_labels, valid_pixels=None):
    if valid_pixels is not None:
        correct_pixels = torch.sum((gt_labels[valid_pixels] == pred_labels[valid_pixels]))
        total_pixels = torch.sum(valid_pixels)
    else:
        correct_pixels = torch.sum(gt_labels == pred_labels)
        total_pixels = torch.prod(torch.tensor(gt_labels.shape))

    accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    return accuracy.item()

def calculate_class_accuracy(gt_labels, pred_labels, class_id, valid_pixels=None):
    if valid_pixels is not None:
        correct_pixels = torch.sum((gt_labels[valid_pixels] == class_id) & (gt_labels[valid_pixels] == pred_labels[valid_pixels]))
        total_gt_pixels = torch.sum(gt_labels[valid_pixels] == class_id)
    else:
        correct_pixels = torch.sum((gt_labels == class_id) & (pred_labels == class_id))
        total_gt_pixels = torch.sum(gt_labels == class_id)

    accuracy = correct_pixels / total_gt_pixels if total_gt_pixels > 0 else 0
    return accuracy.item()

def calculate_class_iou(gt_labels, pred_labels, class_id, valid_pixels=None):
    if valid_pixels is not None:
        intersection = torch.sum((gt_labels[valid_pixels] == class_id) & (pred_labels[valid_pixels] == class_id))
        union = torch.sum((gt_labels[valid_pixels] == class_id) | (pred_labels[valid_pixels] == class_id))
    else:
        intersection = torch.sum((gt_labels == class_id) & (pred_labels == class_id))
        union = torch.sum((gt_labels == class_id) | (pred_labels == class_id))

    iou = intersection / union if union > 0 else 0
    return iou.item()

def calculate_metrics(gt_labels, pred_labels, num_classes, ignore_class=None):
    valid_pixels = None
    if ignore_class is not None:
        valid_pixels = (gt_labels != ignore_class)
    overall_accuracy = calculate_accuracy(gt_labels, pred_labels, valid_pixels=valid_pixels)

    class_accuracies = []
    class_ious = []

    for class_id in range(num_classes):
        class_accuracy = calculate_class_accuracy(gt_labels, pred_labels, class_id, valid_pixels=valid_pixels)
        class_iou = calculate_class_iou(gt_labels, pred_labels, class_id, valid_pixels=valid_pixels)
        class_accuracies.append(class_accuracy)
        class_ious.append(class_iou)

    return overall_accuracy, class_accuracies, class_ious

def main(args):
    # Use the specified dataset configuration
    if args.dataset.lower() == 'cityscapes':
        from cityscapes_config import CityscapesDatasetConfig as DatasetConfig
    elif args.dataset.lower() == 'camvid':
        from camvid_config import CamVidDatasetConfig as DatasetConfig
    else:
        raise ValueError("Invalid dataset. Supported options: 'cityscapes', 'camvid'")
    CLASSES = DatasetConfig.CLASSES
    PALETTE = DatasetConfig.PALETTE
    assert(len(CLASSES) == len(PALETTE))
    print(f"Number of classes: {len(CLASSES)}")

    gt_files = get_images_with_suffix(DatasetConfig.PATH, DatasetConfig.SUFFIX)
    pred_files = get_images_with_suffix(args.pred_path, args.pred_suffix)
    assert(len(gt_files) == len(pred_files))
    print(f"Number of images for evaluation: {len(gt_files)}")
    for file1, file2 in zip(gt_files, pred_files):
        basename1 = os.path.basename(file1).replace(DatasetConfig.SUFFIX, "")
        basename2 = os.path.basename(file2).replace(args.pred_suffix, "")
        assert(basename1 == basename2)

    # Create PyTorch datasets
    gt_dataset = GtDataset(gt_files)
    pred_dataset = PredDataset(pred_files, PALETTE)

    # Create PyTorch data loaders
    gt_loader = DataLoader(gt_dataset, batch_size=1, num_workers=16, shuffle=False)
    pred_loader = DataLoader(pred_dataset, batch_size=1, num_workers=32, shuffle=False)

    # Load ground truth images using DataLoader
    print(f"Loading ground truth masks from {os.path.abspath(DatasetConfig.PATH)}")
    ground_truth_all_torch = torch.cat(list(tqdm(gt_loader, total=len(gt_loader), desc="Processing")))

    # Load predicted label images using DataLoader
    print(f"Loading predicted masks from {os.path.abspath(args.pred_path)}")
    predicted_labels_all_torch = torch.cat(list(tqdm(pred_loader, total=len(pred_loader), desc="Processing")))

    # Ensure the shape of the tensors is correct
    print("Shape of PyTorch ground truth tensor:", ground_truth_all_torch.shape)
    print("Shape of PyTorch predicted labels tensor:", predicted_labels_all_torch.shape)

    num_classes = len(CLASSES)
    overall_accuracy, class_accuracies, class_ious = calculate_metrics(ground_truth_all_torch, predicted_labels_all_torch, num_classes, ignore_class=DatasetConfig.IGNORE_CLASS)

    print(f"aAcc: {overall_accuracy}")
    print(f"mIoU: {sum(class_ious) / num_classes}")
    print(f"mAcc: {sum(class_accuracies) / num_classes}")
    for class_id in range(num_classes):
        print(f"IoU.{CLASSES[class_id]}: {class_ious[class_id]}")
    for class_id in range(num_classes):
        print(f"Acc.{CLASSES[class_id]}: {class_accuracies[class_id]}")

    evaluation_results = dict()
    evaluation_results["dataset"] = args.dataset.lower()
    evaluation_results["path"] = os.path.abspath(args.pred_path)
    evaluation_results["aAcc"] = overall_accuracy
    evaluation_results["mIoU"] = sum(class_ious) / num_classes
    evaluation_results["mAcc"] = sum(class_accuracies) / num_classes
    for class_id in range(num_classes):
        evaluation_results[f"IoU.{CLASSES[class_id]}"] = class_ious[class_id]
    for class_id in range(num_classes):
        evaluation_results[f"Acc.{CLASSES[class_id]}"] = class_accuracies[class_id]

    # Construct the timestamp for the output file
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    json_output_path = f"evaluation_results_{timestamp}.json"

    # Save the results to a JSON file
    with open(json_output_path, "w") as json_file:
        json.dump(evaluation_results, json_file, indent=2)
    print(f"Evaluation results saved to {os.path.abspath(json_output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation Script")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name. Supported options: 'cityscapes', 'camvid'")
    parser.add_argument("--pred_path", type=str, default="./work_dirs/test_cityscapes", help="Path to predicted images")
    parser.add_argument("--pred_suffix", type=str, default="_leftImg8bit.png", help="Predicted image suffix")

    args = parser.parse_args()
    main(args)