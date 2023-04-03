import torch
import clip
# from sklearn.decomposition import PCA
from mmseg.datasets import build_dataset


def get_dataset_classes(dataset_cfg):
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
        type=dataset_cfg['dataset_type'],
        data_root=dataset_cfg['data_root'],
        img_dir=dataset_cfg['img_dir'],
        ann_dir=dataset_cfg['ann_dir'],
        pipeline=train_pipeline
    )

    datasets = [build_dataset(cfg)]

    x = datasets[0].prepare_train_img(0)
    print(x['gt_semantic_seg'])

    return list(datasets[0].CLASSES)


def get_text_embeddings(dataset_cfg, model_name):
    class_names = get_dataset_classes(dataset_cfg)
    model, _ = clip.load(model_name, device='cpu')
    texts = torch.cat([clip.tokenize(c).to('cpu') for c in class_names])
    return model.encode_text(texts)


def perform_pca(text_embeddings, num_class):
    x = text_embeddings.detach().numpy()
    pca = PCA(n_components=num_class)
    x_pca = pca.fit_transform(x)
    return torch.from_numpy(x_pca)


if __name__ == "__main__":
    cfg = dict(
        city=dict(
            dataset_type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            num_class=19,
        ),
        ade20k=dict(
            dataset_type='ADE20KDataset',
            data_root='data/ade/ADEChallengeData2016',
            img_dir='images/training',
            ann_dir='annotations/training',
            num_class=150,
        ),
        camvid=dict(
            dataset_type='CamVidDataset',
            data_root='data/CamVid/',
            img_dir='train',
            ann_dir='train_labelIds',
            num_class=11,
        )
    )

    dataset = 'city'
    model_name = 'RN50'
    dataset_cfg = cfg[dataset]

    text_embeddings = get_text_embeddings(dataset_cfg, model_name)
    model_name = model_name.replace("/", "")
    K, C = text_embeddings.shape
    torch.save(text_embeddings, f"./pretrained/textfeat_{dataset}_{K}_{model_name}_{C}.pth")

    # text_embeddings_pca = perform_pca(text_embeddings, dataset_cfg['num_class'])
    # K, C = text_embeddings_pca.shape
    # torch.save(text_embeddings_pca, f"./pretrained/textfeat_{dataset}_{K}_{model_name}_{C}.pth")
