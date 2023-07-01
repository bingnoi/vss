# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips
from mmseg.models.utils import *
import attr

from IPython import embed

import cv2
from .utils.utils import save_cluster_labels
import time
from torch.nn import functional as F
from .segformer_head import *
from .segformer_clip import *

@HEADS.register_module()
class SegFormerHead_clips(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips, self).__init__(input_transform='multiple_select', **kwargs)
        self.net = SegFormerHead_clipsNet()

    def forward(self, inputs, batch_size=None, num_clips=None):
        #每隔四帧做一次预测，每隔5帧交互生成一次memory_feature
        # if self.mode == 'TRAIN':
        #     for i,frame in enumerate(self.nums_frames):
        #         if i == 0:
        #             self.net(mode='init_memory',inputs=inputs)
        #         if i%self.num_infer==0:
        #             self.net(mode ='update_memory',inputs=inputs)
        #         out = self.net(mode='segment',inputs=inputs)
        # return out

        if self.mode == 'TRAIN':
            for i,frame in enumerate(self.nums_frames):
                if i == 0:
                    self.net(mode='init_memory',inputs=inputs)
                if i%self.num_infer==0:
                    self.net(mode ='update_memory',inputs=inputs)
                out = self.net(mode='segment',inputs=inputs)
        return out