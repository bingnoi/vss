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

from .memory import *

@HEADS.register_module()
class SegFormerHead_clips(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips, self).__init__(input_transform='multiple_select', **kwargs)
        self.net = SegFormerHead_clipsNet(feature_strides = feature_strides,**kwargs)
        self.memory = FeatureMemory()
        

    def forward(self, inputs,batch_size, num_clips,memory):
        #每隔四帧做一次预测，每隔5帧交互生成一次memory_feature

        # if self.mode == 'TRAIN':
        # print('1len ',len(inputs))
        
        num_infer = self.num_infer
        
        # print(inputs[0].shape)
        # n,c,_,_ = inputs[0].shape
        # torch.Size([4, 64, 120, 216])
        
        frame_len = inputs[0].shape[0]
        
        frame_gap = 2
        frame_gap_l = frame_gap*3+1
        
        out_to = []
        
        print(frame_len,frame_gap_l)
        # 4 7
        
        if self.training:
            for i in range(frame_len):
                if i == 0:
                    if memory == None:
                        memoryFeature = self.memory(mode='init_memory',feats=inputs[-1][:5,:])
                    else:
                        #print('mttttt',memoryback.shape)
                        memoryFeature = self.memory(mode='set_memory',feats=memory)
                elif i%num_infer==0 and i>num_infer:
                    num_digit = i / 10
                    num_decimal = i % 10
                    carry = 1 if num_decimal>=5 else 0
                    # print('qqqqq ',inputs[-1].shape,num_digit*10+carry*5-5,num_digit*10+carry*5)
                    memoryFeature = self.memory(mode ='update_memory',feats=inputs[-1][int(num_digit*10+carry*5-5):int(num_digit*10+carry*5),:])
                #print('ss ',i,memoryFeature.shape)
                if i >= frame_gap_l:
                    frame_in = []
                    for sh in range(4):
                        frame_in.append(torch.stack([inputs[sh][q,:] for q in range(i-frame_gap_l,i,frame_gap)],dim=0))
                    # print("memorysssssss",i,memoryFeature.shape)
                    # if memoryFeature == None:
                    #     print(i)
                    out = self.net(mode='segment',feats=memoryFeature,inputs=frame_in,batch_size=1,num_clips=4)
                    # print('see ',i)
                    out_to.append(out)
        else:
            memoryFeature = []
            out = self.net(mode='segment',feats=memoryFeature,inputs=inputs,batch_size=1,num_clips=frame_len)
            out_to.append(out)
        
        out = torch.cat(out_to,dim=1)
        # print('o',out.shape)
        # print('memory',memoryFeature.shape)
        return out,memoryFeature