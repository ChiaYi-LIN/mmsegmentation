import os
import glob
from PIL import Image
from tqdm import tqdm

def get_images_with_suffix(directory, suffix):
    pattern = os.path.join(directory, f'**/*{suffix}')
    image_files = sorted(glob.glob(pattern, recursive=True))
    return image_files

def concatenate_images(input_files, gt_files, baseline_files, ours_files, output_directory):
    # Loop over each set of corresponding images
    for input_file, gt_file, baseline_file, ours_file in tqdm(zip(input_files, gt_files, baseline_files, ours_files), total=len(input_files), desc="Processing"):
        # Open each image
        input_image = Image.open(input_file)
        gt_image = Image.open(gt_file)
        baseline_image = Image.open(baseline_file)
        ours_image = Image.open(ours_file)

        # Concatenate the images horizontally (from left to right)
        new_image = Image.new('RGB', (input_image.width * 4, input_image.height))
        new_image.paste(input_image, (0, 0))
        new_image.paste(gt_image, (input_image.width, 0))
        new_image.paste(baseline_image, (input_image.width * 2, 0))
        new_image.paste(ours_image, (input_image.width * 3, 0))

        # Save the concatenated image
        output_file = os.path.join(output_directory, os.path.basename(input_file))
        new_image.save(output_file)

input_path = "/tmp2/linchiayi/test_exp_env/mmsegmentation/data/cityscapes/leftImg8bit/val"
input_suffix = "_leftImg8bit.png"
input_files = get_images_with_suffix(input_path, input_suffix)

gt_path = "/tmp2/linchiayi/test_exp_env/mmsegmentation/data/cityscapes/gtFine/val"
gt_suffix = "_gtFine_color.png"
gt_files = get_images_with_suffix(gt_path, gt_suffix)

baseline_path = "/tmp2/linchiayi/test_exp_env/mmsegmentation/work_dirs/STDCSeg results on cityscapes png 72.2"
baseline_suffix = "_rgb.png"
baseline_files = get_images_with_suffix(baseline_path, baseline_suffix)

ours_path = "/tmp2/linchiayi/test_exp_env/mmsegmentation/work_dirs/test_cityscapes"
ours_suffix = "_leftImg8bit.png"
ours_files = get_images_with_suffix(ours_path, ours_suffix)

output_directory = "/tmp2/linchiayi/test_exp_env/mmsegmentation/work_dirs/cityscapes_combined"
os.makedirs(output_directory, exist_ok=True)
concatenate_images(input_files, gt_files, baseline_files, ours_files, output_directory)