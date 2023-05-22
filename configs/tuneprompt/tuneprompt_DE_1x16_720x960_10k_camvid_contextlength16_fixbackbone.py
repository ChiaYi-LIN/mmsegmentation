_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/camvid.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]

# model
checkpoint = './work_dirs/detextnet_stdc1_1x16_720x960_10k_camvid/best.pth'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        type='STDCDeContextNet',
        textencoder_cfg=dict(
            type='CLIPTextContextEncoder',
            context_length=16,
            encoder_type='RN50',
            pretrained='./pretrained/RN50.pt'),
        context_mode="CSC",
        CLASSES=('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian',
                 'Road', 'Sidewalk', 'SignSymbol', 'Sky', 'Tree')),
    decode_head=dict(
        in_channels=256+11,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=690000)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=11,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=690000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=11,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=690000),
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
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]),
        dict(
            type='VanillaHead',
            temperature=0.07,
            in_channels=11,
            channels=1,
            num_classes=11,
            in_index=4,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=690000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ]
)

# dataset
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
)

# schedule
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005,
                 paramwise_cfg=dict(
                    custom_keys={
                        'backbone.backbone': dict(lr_mult=0.),
                        'backbone.text_encoder': dict(lr_mult=0., decay_mult=0.),
                        'backbone.contexts': dict(decay_mult=0.),
                        '.bn.': dict(decay_mult=0.)}))

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear', warmup_iters=200, warmup_ratio=1e-5)
