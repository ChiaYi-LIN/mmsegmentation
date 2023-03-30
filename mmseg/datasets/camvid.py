# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamVidDataset(CustomDataset):
    """CamVid dataset.

    The ``img_suffix`` and ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('Bicyclist', 'Building', 'Car', 'Column_Pole', 'Fence', 'Pedestrian',
               'Road', 'Sidewalk', 'SignSymbol', 'Sky', 'Tree')

    PALETTE = [[0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
               [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
               [192, 128, 128], [128, 128, 128], [128, 128, 0]]

    def __init__(self, **kwargs):
        super(CamVidDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_L.png',
            **kwargs)
