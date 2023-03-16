import torch
import clip
from mmseg.datasets import build_dataset


def get_dataset_classes(dataset_type='ADE20KDataset'):
    data_root = 'data/ADEChallengeData2016'
    IMG_MEAN = [v*255 for v in [0.48145466, 0.4578275, 0.40821073]]
    IMG_VAR = [v*255 for v in [0.26862954, 0.26130258, 0.27577711]]

    img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)
    crop_size = (640, 640)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='Resize', img_scale=(2048, 640), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg = dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline
    )

    datasets = [build_dataset(cfg)]
    return list(datasets[0].CLASSES)


def get_text_embeddings(dataset_type, which_clip):
    class_names = get_dataset_classes(dataset_type)
    model, _ = clip.load(which_clip, device='cpu')
    texts = torch.cat([clip.tokenize(c).to('cpu') for c in class_names])
    return model.encode_text(texts)


if __name__ == "__main__":
    dataset_type = 'ADE20KDataset'
    dataset_short = 'ade20k'
    which_clip = "RN50"
    text_embeddings = get_text_embeddings(dataset_type=dataset_type, which_clip=which_clip)
    K, C = text_embeddings.shape
    torch.save(text_embeddings, f"./pretrained/textfeat_{dataset_short}_{K}_{which_clip}_{C}.pth")
