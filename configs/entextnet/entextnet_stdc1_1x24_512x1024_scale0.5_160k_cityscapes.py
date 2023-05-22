_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/cityscapes_0.5scale.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# model
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc1_20220308-5368626c.pth'  # noqa
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='STDCEnTextNet',
        last_in_channels=(1024+19, 512),
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
        text_embeddings='./pretrained/textfeat_city_19_RN50_1024.pth'),
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=780000)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=780000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=780000),
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
            in_channels=19,
            channels=1,
            num_classes=19,
            in_index=4,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=780000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ]
)

# dataset
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=4,
)

# schedule
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005,
                 paramwise_cfg=dict(
                    custom_keys={
                        'backbone.backbone': dict(lr_mult=0.1),
                        'backbone.text_encoder': dict(lr_mult=0., decay_mult=0.),
                        '.bn.': dict(decay_mult=0.)}))

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear', warmup_iters=1000, warmup_ratio=1e-5)
