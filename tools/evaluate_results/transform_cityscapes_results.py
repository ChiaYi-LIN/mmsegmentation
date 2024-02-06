import os
import argparse
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from cityscapes_config import CityscapesDatasetConfig as DatasetConfig

def get_images_with_suffix(directory, suffix):
    pattern = os.path.join(directory, f'**/*{suffix}')
    image_files = sorted(glob.glob(pattern, recursive=True))
    return image_files

def map_rgb_to_id(rgb_image, palette, id_list):
    mapping = {tuple(color): id_val for color, id_val in zip(palette, id_list)}
    mapped_image = np.zeros_like(rgb_image[:, :, 0], dtype=np.uint8)
    for color, id_val in mapping.items():
        mask = np.all(rgb_image == np.array(color), axis=-1)
        mapped_image[mask] = id_val
    return mapped_image

def main(args):
    CLASSES = DatasetConfig.CLASSES
    PALETTE = DatasetConfig.PALETTE
    ID = DatasetConfig.ID

    pred_files = get_images_with_suffix(args.pred_path, args.pred_suffix)

    output_directory = args.output_path
    os.makedirs(output_directory, exist_ok=True)

    for pred_file in tqdm(pred_files, desc="Processing"):
        image = np.array(Image.open(pred_file))
        mapped_image = map_rgb_to_id(image, PALETTE, ID)
        output_file = os.path.join(output_directory, os.path.basename(pred_file))
        Image.fromarray(mapped_image, mode='L').save(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Evaluation Script")
    parser.add_argument("--pred_path", type=str, default="./work_dirs/test_cityscapes", help="Path to predicted images")
    parser.add_argument("--pred_suffix", type=str, default="_leftImg8bit.png", help="Predicted image suffix")
    parser.add_argument("--output_path", type=str, default="./output_grayscale", help="Path to save grayscale images")

    args = parser.parse_args()
    main(args)