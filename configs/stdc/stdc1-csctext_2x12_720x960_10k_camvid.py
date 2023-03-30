_base_ = './stdc1-uctext_2x12_720x960_10k_camvid.py'
model = dict(
    backbone=dict(
        context_mode="CSC",
    ),
)
