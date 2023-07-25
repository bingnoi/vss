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
from .hypercorre import hypercorre_topk2
from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
from torch.nn import functional as F

import math

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# @HEADS.register_module()
# class SegFormerHead(BaseDecodeHead):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, feature_strides, **kwargs):
#         super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
#         assert len(feature_strides) == len(self.in_channels)
#         assert min(feature_strides) == feature_strides[0]
#         self.feature_strides = feature_strides

#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

#         decoder_params = kwargs['decoder_params']
#         embedding_dim = decoder_params['embed_dim']


#         self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

#         self.linear_fuse = ConvModule(
#             in_channels=embedding_dim*4,
#             out_channels=embedding_dim,
#             kernel_size=1,
#             # norm_cfg=dict(type='SyncBN', requires_grad=True)
#             norm_cfg=dict(type='GN', num_groups=1)
#         )

#         self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

#     def encode_key(self, frame, need_sk=True, need_ek=True): 
#         return

#     def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
#         return
    
#     def update_memory(self, query_key, query_selection, memory_key, 
#                     memory_shrinkage, memory_value):
#         return

#     def forward(self, inputs):
#         x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
#         c1, c2, c3, c4 = x

#         print(c1.shape, c2.shape, c3.shape, c4.shape)

#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape

#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

#         x = self.dropout(_c)
#         x = self.linear_pred(x)

#         # print(torch.cuda.memory_allocated(0))

#         return x

class pooling_mhsa(nn.Module):
    def __init__(self,dim):
        super().__init__()

        # pool_ratios=[1,2,3,4]
        # pool_ratios=[12, 16, 20, 24]

        pool_ratios = [[1,2,3,4], [2,4,6,8], [4,8,12,16], [8,16,24,32]]
        # self.pool_ratios = pool_ratios

        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

        self.d_convs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pool_ratios])

        # self.d_convs1 = nn.ModuleList([nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pool_ratios[0]])
        # self.d_convs2 = nn.ModuleList([nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pool_ratios[1]])
        # self.d_convs3 = nn.ModuleList([nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pool_ratios[2]])
        # self.d_convs4 = nn.ModuleList([nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pool_ratios[3]])

    def forward(self,x,redu_ratios):

        B, N, C, hy, wy = x.shape

        pools = []

        x = x.reshape(-1,C,hy,wy)

        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # hy_final = round(hy/redu_ratios[0])+round(hy/redu_ratios[1])+round(hy/redu_ratios[2])+round(hy/redu_ratios[3]) #8,3,4,5
        # wy_final = round(wy/redu_ratios[0])+round(wy/redu_ratios[1])+round(wy/redu_ratios[2])+round(wy/redu_ratios[3])

        # print("hy_wy ",hy_final,wy_final)

        for (l,redu_ratio) in zip(self.d_convs,redu_ratios):

            pool = F.adaptive_avg_pool2d(x, (round(hy/redu_ratio), round(wy/redu_ratio)))

            # print("rrr  ",round(hy/redu_ratio),round(wy/redu_ratio))

            pool = pool + l(pool) # fix backward bug in higher torch versions when training

            pools.append(pool.view(B*N, C, -1))
        
        # print("pp ",pools[0].shape,pools[1].shape,pools[2].shape,pools[3].shape)
        
        pools = torch.cat(pools, dim=2)

        pools = self.norm(pools.permute(0,2,1))#B,-1,C

        # print("pools ",pools.shape)

        pools = pools.reshape(B,N,-1,C)
        
        return pools
        
def softmax_w_top(x, top=20):
    print('x',x.shape)
    x = x.squeeze(1)
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x

@HEADS.register_module()
class SegFormerHead_clipsNet(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides,dim=[64, 128, 320, 512], **kwargs):
        super(SegFormerHead_clipsNet, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.dim=dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.embeding = embedding_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.pooling_mhsa_c1 = pooling_mhsa(c1_in_channels)
        self.pooling_mhsa_c2 = pooling_mhsa(c2_in_channels)
        self.pooling_mhsa_c3 = pooling_mhsa(c3_in_channels)
        self.pooling_mhsa_c4 = pooling_mhsa(c4_in_channels)

        self.pooling_linear = nn.Linear(embedding_dim*4,embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.memory_module = FeatureMemory()

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        self.linear1 = nn.Linear(dim[3],dim[3],bias=True)
        self.linear2 = nn.Linear(dim[3],dim[3],bias=True)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,512, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,512, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,512, self.num_classes)

        # self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        # self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        # self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        self.hypercorre_module=hypercorre_topk2(dim=self.in_channels, backbone=self.backbone)
        
        self.refine_block = nn.Sequential(
            # nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1, bias=False),
            nn.LayerNorm(512)
        )

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True
    
    def forward(self, mode, *args, **kwargs):
        if mode == 'init_memory':
            self.init_memory(*args, **kwargs)
        elif mode == 'update_memory':
            self.update_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.forward_features(*args, **kwargs)
        else:
            raise NotImplementedError
    
    # def init_memory(self,):
    #     B,num_clips,cx,hx,wx=query_frame[0].shape
    #     query_frame_selected = query_frame[0].permute(0,1,3,4,2).reshape(B,num_clips,-1,cx)

    #     query_frame_selected = self.linear1(query_frame_selected)
    #     memory_feature = self.linear2(self.memory.data.permute(0,1,3,4,2)).reshape(B,num_clips,-1,cx)

    #     # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
    #     atten = torch.matmul(memory_feature,query_frame_selected.transpose(-1,-2))

    #     #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
    #     out = torch.matmul(atten,query_frame_selected).reshape(B,num_clips,hx,wx,cx).permute(0,1,4,2,3)
        
    #     self.memory.data = out
    #     query_frame[0] = out

    # def update_memory(self,inputs):
    #     B,num_clips,cx,hx,wx=query_frame[0].shape
    #     query_frame_selected = query_frame[0].permute(0,1,3,4,2).reshape(B,num_clips,-1,cx)

    #     query_frame_selected = self.linear1(query_frame_selected)
    #     memory_feature = self.linear2(self.memory.data.permute(0,1,3,4,2)).reshape(B,num_clips,-1,cx)

    #     # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
    #     atten = torch.matmul(memory_feature,query_frame_selected.transpose(-1,-2))

    #     #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
    #     out = torch.matmul(atten,query_frame_selected).reshape(B,num_clips,hx,wx,cx).permute(0,1,4,2,3)
        
    #     self.memory.data = out
    #     query_frame[0] = out
    
    
    def global_matching(self,mk,qk):
        B, CK, _,_ = mk.shape
        
        mk = mk.flatten(start_dim=2)
        # print('q',qk.shape)
        qk = qk.flatten(start_dim=3)

        a = mk.pow(2).sum(1).unsqueeze(2)
        
        # print(mk.transpose(1, 2).shape,qk.shape)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # We don't actually need this, will update paper later
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a+b) / math.sqrt(CK)  # B, NE, HW
        affinity = softmax_w_top(affinity,top=20)  # B, THW, HW

        return affinity
    
    def read_out(self,affinity,mv):
        return torch.bmm(mv, affinity)

    def forward_features(self,feats,inputs, batch_size=None, num_clips=None):
        #每一层特征下做down_sample,按通道cancat,每一次s*c->s*4c
        # print('a ',len(inputs))
        # print('b ',inputs[0].shape)
        
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        
        # print('shape2',[i.shape for i in inputs])
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        
        # print('c40 ',c4.shape)

        ############## MLP decoder on C1-C4 ###########
        # print(c4.shape)
        # torch.Size([2048, 15, 15])
        
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) 

        #downsample to c1(512,512)

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        # print('c41 ',c4.shape)
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        query_frame=[query_c1, query_c2, query_c3, query_c4]
        
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
        supp_c1,supp_c2,supp_c3,supp_c4 = supp_frame[0],supp_frame[1],supp_frame[2],supp_frame[3] 
        
        # query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        # query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # query_c1=self.sr1(query_c1)
        # query_c2=self.sr2(query_c2)
        # query_c3=self.sr3(query_c3)

        # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])

        # query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        # query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        
        # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])
        
        B,num_clips_select,cx,hx,wx=supp_c4.shape
        
        # store_c4 = supp_c4
        # print('c42 ',query_c4.shape)
        
        if len(feats)>0:
            supp_frame_selected = supp_c4.permute(0,1,3,4,2)
            # print('ss ',query_c4[:,i].shape)
            supp_frame_selected = self.linear1(supp_frame_selected.reshape(B,num_clips_select*hx*wx,cx)) #b,-1,cx
            
            # B,cx,hx,wx=feats.shape 
            B,cx=feats.shape 
            
            # memory_feature = self.linear2(feats.permute(0,2,3,1).reshape(B,hx*wx,cx))
            memory_feature = self.linear2(feats)
            
            atten = torch.matmul(supp_frame_selected,memory_feature.transpose(-1,-2)) #b,h,w,c c,b = b,h,w,b * b,c 

            #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
            # print('ss',supp_frame_selected.shape,memory_feature.transpose(-1,-2).shape)
            # print('ts',torch.matmul(atten,memory_feature).shape)
            
            supp_c4 = torch.matmul(atten,memory_feature).reshape(B,hx,wx,cx).permute(0,3,1,2)
            
            # affinity = self.global_matching(feats,supp_c4)
            # readout_mem = self.read_out(affinity.expand(k,-1,-1),feats)
            # print('tes',readout_mem.shape)
            # exit()
            # print('f1 ',supp_c4.shape)
            # supp_c4 = self.refine_block(supp_c4)
            # print('f2 ',supp_c4.shape)
            # exit()
            
            supp_c4 = supp_c4.unsqueeze(1)
            
            # supp_c4 = supp_c4 + store_c4
        
        
        supp_frame = [supp_c1, supp_c2, supp_c3, supp_c4]
        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print('check1',[i.shape for i in query_frame])
        # print('check2',[i.shape for i in supp_frame])

        final_feature = self.hypercorre_module(query_frame,supp_frame) 
        

        ####生成attention——weight

        supp_feats = final_feature

        # atten=self.hypercorre_module(query_frame, supp_frame)
        # atten=F.softmax(atten,dim=-1)

        #(B,N,-1,C)
        # print("atten shape",atten.shape)
        # shape torch.Size([1, 3, 3600, 256])


        # pooling_c1 = self.pooling_mhsa_c1(c1[:,:-1],[8,16,24,32])
        # pooling_c2 = self.pooling_mhsa_c2(c2[:,:-1],[4,8,12,16])
        # pooling_c3 = self.pooling_mhsa_c3(c3[:,:-1],[2,4,6,8])  
        # pooling_c4 = self.pooling_mhsa_c4(c4[:,:-1],[1,2,3,4])  #<----

        # print("c1 c2",c1[:,:-1].shape,c2[:,:-1].shape,c3[:,:-1].shape,c4[:,:-1].shape)
        # torch.Size([1, 3, 64, 120, 120]) torch.Size([1, 3, 128, 60, 60]) torch.Size([1, 3, 320, 30, 30]) torch.Size([1, 3, 512, 15, 15])

        # print("pooling",pooling_c1.shape,pooling_c2.shape,pooling_c3.shape,pooling_c4.shape)
        # torch.Size([1, 3, 256, 64]) torch.Size([1, 3, 256, 128]) torch.Size([1, 3, 256, 320]) torch.Size([1, 3, 256, 512])

        # [torch.Size([2, 1, 64, 120, 120]), torch.Size([2, 1, 128, 60, 60]), torch.Size([2, 1, 320, 30, 30]), torch.Size([2, 1, 512, 15, 15])]

        # pooling_all_scale = torch.cat((pooling_c1,pooling_c2,pooling_c3,pooling_c4),dim=3) # B,N,-1,4C

        # print('scale',pooling_all_scale.shape)
        # torch.Size([1, 3, 256, 1024])

        # down_pooling_all_scale = self.pooling_linear(pooling_all_scale) #B,N,-1,C --- 1,4,130,256

        # down_pooling_all_scale = down_pooling_all_scale.reshape(batch_size,num_clips-1,-1,self.embeding)

        # down_pooling_all_scale = down_pooling_all_scale.transpose(-1,-2)
        

        h2=int(h/2)
        w2=int(w/2)
        # # h3,w3=shape_c3[-2], shape_c3[-1]
        
        
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        
        # print('c21',_c2.shape,batch_size,num_clips,h2,w2)
        
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)

        # # _c_further=_c2[:,:-1].reshape(batch_size, num_clips-1, -1, h3*w3)
        # _c3=self.sr1_feat(_c2)
        # _c3=_c3.reshape(batch_size, num_clips, -1, _c3.shape[-2]*_c3.shape[-1]).transpose(-2,-1)
        # # _c_further=_c3[:,:-1].reshape(batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], -1)    ## batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], c
        # _c_further=_c3[:,:-1]        ## batch_size, num_clips-1, _c2.shape[-2]*_c2.shape[-1], c
        # # print(_c_further.shape, topk_mask.shape, torch.unique(topk_mask.sum(2)))
        # _c_further=_c_further[topk_mask].reshape(batch_size,num_clips-1,-1,_c_further.shape[-1])    ## batch_size, num_clips-1, s, c
        # supp_feats=torch.matmul(atten,_c_further)#qk*v

        # from here !!!!!!

        # supp_feats=supp_feats.reshape(batch_size, (num_clips-1), self.embeding, -1)

        # for i in range(0,4):
        #     supp_feats[i]=supp_feats[i].transpose(-2,-1).reshape(batch_size,-1,self.embeding,h2,w2)

        supp_feats = [ supp.transpose(-2,-1).reshape(batch_size,-1, self.embeding, h2,w2) for supp in supp_feats ]
        
        new_supp = []

        for i in range(0,3):
            new_supp.append(torch.cat([supp_feats[i],_c2_split[:,i+1:i+2]],dim=2))

        supp_feats = new_supp

        supp_feats.insert(0,_c2_split[:,0])

        supp_feats=[ii.squeeze(1) for ii in supp_feats]

        outs=supp_feats
        

        # ends here !!!!!!

        # print('p',c1[:,0].shape,c2[:,0].shape,c3[:,0].shape,c4[:,0].shape)

        # from here!!!

        # pooling_c1 = self.pooling_mhsa_c1(c1[:,0].unsqueeze(1),[8,16,24,32])
        # pooling_c2 = self.pooling_mhsa_c2(c2[:,0].unsqueeze(1),[4,8,12,16])
        # pooling_c3 = self.pooling_mhsa_c3(c3[:,0].unsqueeze(1),[2,4,6,8])  
        # pooling_c4 = self.pooling_mhsa_c4(c4[:,0].unsqueeze(1),[1,2,3,4])  #<----

        # pooling_ct = torch.cat([pooling_c1,pooling_c2,pooling_c3,pooling_c4],dim=3)

        # pooling_ct = self.pooling_linear(pooling_ct)

        # supp_feats.insert(0,pooling_ct)

        # outs = [i.squeeze(1).permute(0,2,1).reshape(1,self.embeding,1,-1) for i in supp_feats]

        # ends here!!!

        # supp_feats=supp_feats.transpose(-2,-1).reshape(batch_size,(num_clips-1),self.embeding,h2,w2)
        # supp_feats=(torch.chunk(supp_feats, (num_clips-1), dim=1)) #b,channel,-1 * (nums_clip-1)
        # supp_feats=[ii.squeeze(1) for ii in supp_feats]
        # supp_feats.append(_c2_split[:,-1])

        # supp_feats.append(pooling_c4[:,-1])

        # print("shape",[i.shape for i in supp_feats])
        
        # [torch.Size([1, 256, 225]), torch.Size([1, 256, 225]), torch.Size([1, 256, 225])]

        # supp_feats.append(_c2_split[:,-1]) #(batch_size, -1, h2, w2) 60*60

        # outs=[i.reshape(batch_size,self.embeding,h2,w2) for i in outs]

        # print([i.shape for i in outs])
        # [torch.Size([1, 256, 225]), torch.Size([1, 256, 225]), torch.Size([1, 256, 225])]
        
        # out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out2=resize(self.deco2(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out3=resize(self.deco3(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        
        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        # out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w
        # output=torch.cat([x,out4],dim=1)

        # print("test ok")
        # a=input()

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            # print('shape ',out4.squeeze(1).shape)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output
        # return output[:,-1:]

   
class small_decoder2(nn.Module):

    def __init__(self,
                 input_dim=256, hidden_dim=256, num_classes=124,dropout_ratio=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.smalldecoder=nn.Sequential(
            # ConvModule(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            # ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1)
            )
        # self.dropout=
        
    def forward(self, input):

        output=self.smalldecoder(input)

        return output