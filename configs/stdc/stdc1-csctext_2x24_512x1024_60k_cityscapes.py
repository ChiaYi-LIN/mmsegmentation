_base_ = './stdc1-uctext_2x24_512x1024_60k_cityscapes.py'
model = dict(
    backbone=dict(
        context_mode="CSC",
    ),
)
