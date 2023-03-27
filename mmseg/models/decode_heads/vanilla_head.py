import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class VanillaHead(BaseDecodeHead):
    def __init__(self, temperature, **kwargs):
        super(VanillaHead, self).__init__(**kwargs)
        self.temperature = temperature
        self.conv_seg = None

    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        output = x / self.temperature

        return output
