# Setup

## Step 1: Clone the base repository

In your preferred terminal or command prompt, run the following command to clone the base repository:

```bash
git clone https://github.com/sChiaYi-LIN/mmsegmentation.git
```

## Step 2: Create Conda Environment

Open a terminal and create a new Conda environment named rtss_text_vl (or any preferred name) with Python version 3.8.16:
```bash
conda create --name rtss_text_vl python==3.8.16
```

Activate the created Conda environment:
```bash
conda activate rtss_text_vl
```

## Step 3: Install PyTorch

Install PyTorch with the specified versions and dependencies:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

## Step 4: Install MMCV

Install MMCV using openmim:

```bash
pip install -U openmim
mim install mmcv-full==1.7.0
```

## Step 5: Install mmsegmentation

Navigate to the mmsegmentation directory and install the package:

```bash
cd mmsegmentation/
pip install -v -e .
```

## Step 6: Install Other Dependencies

Install additional dependencies:

```bash
pip install timm gdown ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Step 7: Download Datasets

Create a data directory outside the mmsegmentation folder:
```bash
cd ..
mkdir data
cd data
```

Download and extract the Cityscapes dataset:
```bash
wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar
tar -xvf cityscapes.tar
rm cityscapes.tar
```

Download and unzip the CamVid dataset:
```bash
gdown 1w9Hyp0Nx8FzyUBEnQla4DHDnFsCf7vZv
unzip CamVid.zip
rm CamVid.zip
```

## Step 8: Softlink Datasets to Base Repository

Inside the mmsegmentation/data/ directory, create softlinks to the downloaded datasets:

```bash
cd mmsegmentation/
mkdir data
cd data
ln -s ../../data/cityscapes/ cityscapes
ln -s ../../data/CamVid/ CamVid
```

## Step 9: Download Checkpoints

Navigate to the mmsegmentation/pretrained/ directory and run the following commands:

```bash
bash download_clip_models.sh
bash download_checkpoints.sh
```

## Step 10: Run Evaluation for Cityscapes

In the mmsegmentation/ directory, execute the following command for Cityscapes evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/tuneprompt/tuneprompt_EN_1x16_512x1024_scale0.5_160k_cityscapes_contextlength16_fixbackbone.py pretrained/tuneprompt_EN_1x16_512x1024_scale0.5_160k_cityscapes_contextlength16_fixbackbone_iter_160000.pth --work-dir ./work_dirs/test_cityscapes --eval mIoU --show-dir ./work_dirs/test_cityscapes/results --opacity 1
```

Execute the following command for CamVid evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py configs/tuneprompt/tuneprompt_1x16_720x960_10k_camvid_contextlength16_fixbackbone.py pretrained/tuneprompt_1x16_720x960_10k_camvid_contextlength16_fixbackbone_iter_9000.pth --work-dir ./work_dirs/test_camvid --eval mIoU --show-dir ./work_dirs/test_camvid/results --opacity 1
```