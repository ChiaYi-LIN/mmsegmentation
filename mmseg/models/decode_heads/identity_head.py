import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(**kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        output = self._transform_inputs(inputs)

        return output
