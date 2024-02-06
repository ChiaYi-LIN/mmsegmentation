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

def load_from_l_mask(file_path):
    img = Image.open(file_path).convert('L')
    img_array = np.array(img, dtype=np.int32)

    return img_array

def load_from_rgb_mask(file_path, config):
    img = Image.open(file_path).convert('RGB')
    img_array = torch.tensor(np.array(img), dtype=torch.uint8)

    # Map RGB values to class indices based on the palette
    label_array = torch.zeros_like(img_array[:, :, 0], dtype=torch.uint8)
    for idx, color in enumerate(config.PALETTE):
        mask = torch.all(img_array == torch.tensor(color), dim=-1)
        label_array[mask] = config.ID[idx]

    return label_array

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

def calculate_metrics(gt_files, pred_files, num_classes=None, args=None, config=None):
    def one_hot_encode(labels, num_classes):
        one_hot = torch.zeros((labels.shape[0], labels.shape[1], num_classes), dtype=torch.uint8)
        for idx in range(num_classes):
            one_hot[:, :, idx][labels == idx] = 1
        return one_hot
    
    ignore_class = config.IGNORE_CLASS if config.IGNORE_CLASS is not None else None 
    palette = config.PALETTE

    # total_accuracy = 0
    class_correct_pixels = torch.zeros(num_classes, dtype=torch.int64)
    class_total_pixels = torch.zeros(num_classes, dtype=torch.int64)
    class_intersection = torch.zeros(num_classes, dtype=torch.int64)
    class_union = torch.zeros(num_classes, dtype=torch.int64)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    ignore_pixels = 0

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Processing"):
        gt_basename = os.path.basename(gt_file).replace(config.SUFFIX, "")
        pred_basename = os.path.basename(pred_file).replace(args.pred_suffix, "")
        assert(gt_basename == pred_basename)

        gt_labels = load_from_l_mask(gt_file)
        pred_labels = load_from_rgb_mask(pred_file, palette=palette)
        # pred_labels = load_from_l_mask(pred_file)
        
        if ignore_class is not None:
            valid_pixels = (gt_labels != ignore_class)
            ignore_pixels += torch.sum(gt_labels == ignore_class).item()
        else:
            valid_pixels = None

        gt_one_hot = one_hot_encode(gt_labels, num_classes)
        pred_one_hot = one_hot_encode(pred_labels, num_classes)
  
        # Check if there is any information loss during the one-hot encoding process
        gt_classes, gt_counts = torch.unique(gt_labels, return_counts=True)
        gt_counts_extend = torch.zeros(num_classes, dtype=torch.int64)
        for this_class, count in zip(gt_classes, gt_counts):
            if this_class.item() < num_classes:
                gt_counts_extend[this_class.item()] = count
        assert(torch.equal(gt_counts_extend, torch.sum(gt_one_hot, dim=(0, 1))))

        if valid_pixels is not None:
            # print(valid_pixels)
            # valid_pixels_expanded = valid_pixels.unsqueeze(-1)
            # gt_one_hot = gt_one_hot & valid_pixels_expanded
            pred_one_hot = pred_one_hot * valid_pixels.unsqueeze(-1)

        # correct_pixels = torch.sum((gt_one_hot == pred_one_hot) & (gt_one_hot == 1))
        # total_pixels = torch.sum(valid_pixels) if valid_pixels is not None else torch.prod(torch.tensor(gt_labels.shape))
        # accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

        # total_accuracy += accuracy

        # class_correct_pixels += torch.sum((gt_one_hot == pred_one_hot) & (gt_one_hot == 1), dim=(0, 1))
        # class_total_pixels += torch.sum(gt_one_hot == 1, dim=(0, 1))
        # class_intersection += torch.sum((gt_one_hot & pred_one_hot) == 1, dim=(0, 1))
        # class_union += torch.sum((gt_one_hot | pred_one_hot) == 1, dim=(0, 1))
        
        

    class_correct_pixels = torch.diag(confusion_matrix)
    class_total_pixels = torch.sum(confusion_matrix, dim=1)
    ious = class_correct_pixels / (torch.sum(confusion_matrix, dim=0) + torch.sum(confusion_matrix, dim=1) - class_correct_pixels)

    return (torch.sum(class_correct_pixels) / torch.sum(class_total_pixels)).item(), \
           (class_correct_pixels / class_total_pixels).tolist(), \
           (ious).tolist(), \
           (class_total_pixels).tolist(), \
           ignore_pixels, \
           confusion_matrix

def generate_confusion_matrix(config):
    config.evalLabels = []
    for label, id in config.LABELS.items():
        if (id < 0):
            continue
        # we append all found labels, regardless of being ignored
        config.evalLabels.append(id)
    maxId = max(config.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1), dtype=np.ulonglong)

def calculate_confusion_matrix(gt_files, pred_files, config):
    confusion_matrix = generate_confusion_matrix(config)

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Processing"):
        gt_labels = load_from_l_mask(gt_file)
        pred_labels = load_from_l_mask(pred_file)
    
        # Populate confusion matrix
        encoding_value = max(gt_labels.max(), pred_labels.max()).astype(np.int32) + 1
        encoded = (gt_labels.astype(np.int32) * encoding_value) + pred_labels

        values, cnt = np.unique(encoded, return_counts=True)

        for value, c in zip(values, cnt):
            pred_id = value % encoding_value
            gt_id = int((value - pred_id) / encoding_value)
            confusion_matrix[gt_id][pred_id] += c

    return confusion_matrix

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
    # gt_dataset = GtDataset(gt_files)
    # pred_dataset = PredDataset(pred_files, PALETTE)

    # Create PyTorch data loaders
    # gt_loader = DataLoader(gt_dataset, batch_size=1, num_workers=16, shuffle=False)
    # pred_loader = DataLoader(pred_dataset, batch_size=1, num_workers=16, shuffle=False)

    # Load ground truth images using DataLoader
    # print(f"Loading ground truth masks from {os.path.abspath(DatasetConfig.PATH)}")
    # ground_truth_all_torch = torch.cat(list(tqdm(gt_loader, total=len(gt_loader), desc="Processing")))

    # unique_values, counts = torch.unique(ground_truth_all_torch, return_counts=True)
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    # print(f"Total: {torch.sum(counts)}")

    # Load predicted label images using DataLoader
    # print(f"Loading predicted masks from {os.path.abspath(args.pred_path)}")
    # predicted_labels_all_torch = torch.cat(list(tqdm(pred_loader, total=len(pred_loader), desc="Processing")))

    # unique_values, counts = torch.unique(predicted_labels_all_torch, return_counts=True)
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    # print(f"Total: {torch.sum(counts)}")

    # Ensure the shape of the tensors is correct
    # print("Shape of PyTorch ground truth tensor:", ground_truth_all_torch.shape)
    # print("Shape of PyTorch predicted labels tensor:", predicted_labels_all_torch.shape)

    num_classes = len(CLASSES)
    # overall_accuracy, class_accuracies, class_ious, class_total_pixels, ignore_pixels, confusion_matrix = calculate_metrics(gt_files, pred_files, num_classes=num_classes, args=args, config=DatasetConfig)

    confusion_matrix = calculate_confusion_matrix(gt_files, pred_files, DatasetConfig)
    eval_matrix = confusion_matrix[DatasetConfig.ID, :][:, DatasetConfig.ID]

    correct = np.diag(eval_matrix)
    row_sum = np.sum(eval_matrix, axis=1)
    col_sum = np.sum(eval_matrix, axis=0)

    overall_accuracy = np.sum(correct) / np.sum(eval_matrix)
    class_ious = (correct / (row_sum + col_sum - correct)).tolist()
    class_accuracies = (correct / row_sum).tolist()
    class_total_pixels = row_sum.tolist()

    print(f"aAcc: {overall_accuracy}")
    print(f"mIoU: {sum(class_ious) / num_classes}")
    print(f"mAcc: {sum(class_accuracies) / num_classes}")
    for class_id in range(num_classes):
        print(f"IoU.{CLASSES[class_id]}: {class_ious[class_id]}")
    for class_id in range(num_classes):
        print(f"Acc.{CLASSES[class_id]}: {class_accuracies[class_id]}")
    for class_id in range(num_classes):
        print(f"Pixels.{CLASSES[class_id]}: {class_total_pixels[class_id]}")
    # print(f"Pixels.ignore = {ignore_pixels}")

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
    for class_id in range(num_classes):
        evaluation_results[f"Pixels.{CLASSES[class_id]}"] = class_total_pixels[class_id]
    # evaluation_results["Pixels.ignore"] = ignore_pixels

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
    parser.add_argument("--pred_path", type=str, default="./work_dirs/test_cityscapes_labelids", help="Path to predicted images")
    parser.add_argument("--pred_suffix", type=str, default="_leftImg8bit.png", help="Predicted image suffix")

    args = parser.parse_args()
    main(args)