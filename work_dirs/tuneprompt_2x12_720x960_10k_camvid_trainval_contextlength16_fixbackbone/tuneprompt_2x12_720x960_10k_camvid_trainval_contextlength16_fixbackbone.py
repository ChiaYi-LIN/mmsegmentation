norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='STDCContextNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(1035, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
        textencoder_cfg=dict(
            type='CLIPTextContextEncoder',
            context_length=16,
            encoder_type='RN50',
            pretrained='./pretrained/RN50.pt'),
        context_mode='CSC',
        CLASSES=('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence',
                 'Pedestrian', 'Road', 'Sidewalk', 'SignSymbol', 'Sky',
                 'Tree')),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=19,
        in_index=3,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=510000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=11,
            in_index=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=510000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=11,
            in_index=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=510000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]),
        dict(
            type='VanillaHead',
            temperature=0.07,
            in_channels=11,
            channels=1,
            num_classes=11,
            in_index=4,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=510000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        './work_dirs/entextnet_stdc1_2x12_720x960_10k_camvid_trainval/latest.pth'
    ))
dataset_type = 'CamVidDataset'
data_root = 'data/CamVid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (720, 960)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=(960, 720),
        ratio_range=(0.5, 2.5),
        scale_step_size=0.25),
    dict(type='RandomCrop', crop_size=(720, 960), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(720, 960), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='CamVidDataset',
        data_root='data/CamVid/',
        img_dir='trainval',
        ann_dir='trainval_labelIds',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize',
                img_scale=(960, 720),
                ratio_range=(0.5, 2.5),
                scale_step_size=0.25),
            dict(type='RandomCrop', crop_size=(720, 960), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(720, 960), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CamVidDataset',
        data_root='data/CamVid/',
        img_dir='test',
        ann_dir='test_labelIds',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CamVidDataset',
        data_root='data/CamVid/',
        img_dir='test',
        ann_dir='test_labelIds',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 720),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys=dict(
            {
                'backbone.backbone': dict(lr_mult=0.0),
                'backbone.text_encoder': dict(lr_mult=0.0, decay_mult=0.0),
                'backbone.contexts': dict(decay_mult=0.0),
                '.bn.': dict(decay_mult=0.0)
            })))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1e-05)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(
    interval=1000, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint = './work_dirs/entextnet_stdc1_2x12_720x960_10k_camvid_trainval/latest.pth'
work_dir = './work_dirs/tuneprompt_2x12_720x960_10k_camvid_trainval_contextlength16_fixbackbone'
gpu_ids = range(0, 2)
auto_resume = False
