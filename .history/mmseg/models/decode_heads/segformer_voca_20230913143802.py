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

from .zero_shot_predictor import TransformerZeroshotPredictor
from .per_pixel import TransformerEncoderPixelDecoder


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result

def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        return semseg

def semantic_inference2(self, mask_cls, mask_pred):
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    return semseg

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
        


@HEADS.register_module()
class SegFormerHead_ZeroShot(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_ZeroShot, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

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

        self.linear_pred = nn.Conv2d(embedding_dim, 512, kernel_size=1)

        # self.memory_module = Memory()

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,512, 512)
        self.deco2=small_decoder2(embedding_dim,512, 512)
        self.deco3=small_decoder2(embedding_dim,512, 512)
        self.deco4=small_decoder2(embedding_dim,512, 512)

        # self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        # self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        # self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        self.hypercorre_module=hypercorre_topk2(dim=self.in_channels, backbone=self.backbone)

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
        self.predictor = TransformerZeroshotPredictor()
        
        self.linearconv1 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        
        # self.linear1 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        # self.linear2 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        # self.linear3 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        # self.linear4 = nn.Conv2d(256,512,kernel_size=(1, 1), stride=(1, 1))
        
        self.clip_classification = True
        self.ensembling = True
        self.ensembling_all_cls = True
        
        self.pixel_decoder = TransformerEncoderPixelDecoder()

    def forward(self,inputs, batch_size=None, num_clips=None):
        
        # print('a ',len(inputs))
        # print('b ',inputs[0].shape)
        # torch.Size([30, 64, 120, 120])

        #每一层特征下做down_sample,按通道cancat,每一次s*c->s*4c
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        # if inputs[0].shape[0] == 1:
        #     a=input()
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # print(c4.shape)
        # torch.Size([30, 512, 15, 15])
        # a=input()

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

        # print('c ',x.shape)
        # c=input()

        # print(x.shape)
        
        # print(self.training,num_clips,self.num_clips)
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
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        # remove last frame
        
        
        # query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        # query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # query_c1=self.sr1(query_c1)
        # query_c2=self.sr2(query_c2)
        # query_c3=self.sr3(query_c3)

        # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])

        # query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        # query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        
        # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])

        query_frame=[query_c1, query_c2, query_c3, query_c4]

        # print('a',query_c4.shape)
        # torch.Size([1, 3, 512, 15, 15])
        # a=input()

        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print('check1',[i.shape for i in query_frame])
        # print('check2',[i.shape for i in supp_frame])
        # check1 [torch.Size([1, 3, 64, 120, 216]), torch.Size([1, 3, 128, 60, 108]), torch.Size([1, 3, 320, 30, 54]), torch.Size([1, 3, 512, 15, 27])]
        # check2 [torch.Size([1, 1, 64, 120, 216]), torch.Size([1, 1, 128, 60, 108]), torch.Size([1, 1, 320, 30, 54]), torch.Size([1, 1, 512, 15, 27])]
        

        final_feature = self.hypercorre_module(query_frame,supp_frame)  

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
        # c21 torch.Size([4, 256, 60, 60]) 1 4 60 60
        
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
        
        # su = [i.squeeze(1) for i in supp_feats]
        # su.insert(0,_c2_split[:,0])
        # su[0] = self.linear1(su[0])
        # su[1] = self.linear2(su[1])
        # su[2] = self.linear3(su[2])
        # su[3] = self.linear4(su[3])
        # supp_feats_f = torch.cat(su,dim=0)
        
        new_supp =  []
        for i in range(0,3):
            new_supp.append(torch.cat([supp_feats[i],_c2_split[:,i+1:i+2]],dim=2))
        supp_feats = new_supp
        supp_feats.insert(0,_c2_split[:,0])
        supp_feats=[ii.squeeze(1) for ii in supp_feats]
        supp_feats[0] = self.linearconv1(supp_feats[0]) 
        # outs_f = torch.cat(outs)

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
        
        out1=resize(self.deco1(supp_feats[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out2=resize(self.deco2(supp_feats[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out3=resize(self.deco3(supp_feats[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out4=resize(self.deco4(supp_feats[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        # # out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # # out4=resize(self.deco4(outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # # out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        output=torch.cat([x,out1,out2,out3,out4],dim=1).squeeze(0)   ## b*(k+k)*124*h*w
        
        # print("ss",[i.shape for i in supp_feats])
        # output_f = torch.cat(supp_feats,dim=0)
        
        mask_feature,transformer_encoder_feature = self.pixel_decoder(supp_feats)
        
        prediction =self.predictor(transformer_encoder_feature,mask_feature)

        if not self.training:
            return prediction[:,-1:].squeeze(1)
            
            # if self.clip_classification:
            #     ##########################
            #     mask_pred_results_224 = F.interpolate(mask_pred_results,
            #         size=(224, 224), mode="bilinear", align_corners=False,)
            #     images_tensor = F.interpolate(_c,
            #                                   size=(224, 224), mode='bilinear', align_corners=False,)
            #     mask_pred_results_224 = mask_pred_results_224.sigmoid() > 0.5

            #     mask_pred_results_224 = mask_pred_results_224.unsqueeze(2)

            #     print("ss",_c.shape,images_tensor.unsqueeze(1).shape,mask_pred_results_224.shape)
            #     # masked_image_tensors = (images_tensor.unsqueeze(1) * mask_pred_results_224)
            #     cropp_masked_image = True
            #     # vis_cropped_image = True
            #     if cropp_masked_image:
            #         # import ipdb; ipdb.set_trace()
            #         mask_pred_results_ori = mask_pred_results
            #         mask_pred_results_ori = mask_pred_results_ori.sigmoid() > 0.5
            #         mask_pred_results_ori = mask_pred_results_ori.unsqueeze(2)
            #         masked_image_tensors_ori = (_c.unsqueeze(1) * mask_pred_results_ori)
            #         # TODO: repeat the clip_images.tensor to get the non-masked images for later crop.
            #         ori_bs, ori_num_queries, ori_c, ori_h, ori_w = masked_image_tensors_ori.shape
            #         # if vis_cropped_image:
            #         clip_images_repeat = _c.unsqueeze(1).repeat(1, ori_num_queries, 1, 1, 1)
            #         clip_images_repeat = clip_images_repeat.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)

            #         masked_image_tensors_ori = masked_image_tensors_ori.reshape(ori_bs * ori_num_queries, ori_c, ori_h, ori_w)
            #         import torchvision
            #         import numpy as np
            #         # binary_mask_preds: [1, 100, 512, 704]
            #         binary_mask_preds = mask_pred_results.sigmoid() > 0.5
            #         binary_bs, binary_num_queries, binary_H, binary_W = binary_mask_preds.shape
            #         # assert binary_bs == 1
            #         binary_mask_preds = binary_mask_preds.reshape(binary_bs * binary_num_queries,
            #                                                       binary_H, binary_W)
            #         sum_y = torch.sum(binary_mask_preds, dim=1)
            #         cumsum_x = torch.cumsum(sum_y, dim=1).float()
            #         xmaxs = torch.argmax(cumsum_x, dim=1, keepdim=True) # shape: [100, 1]
            #         cumsum_x[cumsum_x==0] = np.inf
            #         xmins = torch.argmin(cumsum_x, dim=1, keepdim=True)
            #         sum_x = torch.sum(binary_mask_preds, dim=2)
            #         cumsum_y = torch.cumsum(sum_x, dim=1).float()
            #         ymaxs = torch.argmax(cumsum_y, dim=1, keepdim=True)
            #         cumsum_y[cumsum_y==0] = np.inf
            #         ymins = torch.argmin(cumsum_y, dim=1, keepdim=True)
            #         areas = (ymaxs - ymins) * (xmaxs - xmins)
            #         ymaxs[areas == 0] = _c.shape[-2]
            #         ymins[areas == 0] = 0
            #         xmaxs[areas == 0] = _c.shape[-1]
            #         xmins[areas == 0] = 0
            #         boxes = torch.cat((xmins, ymins, xmaxs, ymaxs), 1)  # [binary_bs * binary_num_queries, 4]
            #         # boxes = boxes.reshape(binary_bs, binary_num_queries, 4)
            #         # TODO: crop images by boxes in the original image size
            #         # boxes_list = [boxes[i].reshape(1, -1) for i in range(boxes.shape[0])]
            #         boxes_list = []
            #         for i in range(boxes.shape[0]):
            #             boxes_list.append(boxes[i].reshape(1, -1).float())
            #         box_masked_images = torchvision.ops.roi_align(masked_image_tensors_ori, boxes_list, 224, aligned=True)
            #         box_masked_images = box_masked_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

            #         # if vis_cropped_image:
            #             # import ipdb; ipdb.set_trace()
            #         box_images = torchvision.ops.roi_align(clip_images_repeat, boxes_list, 224, aligned=True)
            #         box_images = box_images.reshape(ori_bs, ori_num_queries, ori_c, 224, 224)

            #     count = 0
            # processed_results = []
            # for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            #     mask_cls_results, mask_pred_results, _c, _c.shape[-2:]
            # ):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])

            #     if self.clip_classification:
            #         import numpy as np
            #         masked_image_tensor = masked_image_tensors[count]
            #         # if cropp_masked_image:
            #         box_masked_image_tensor = box_masked_images[count]
            #         # if vis_cropped_image:
            #         box_image_tensor = box_images[count]
            #         # boxs = boxes_list[count]
            #         count = count + 1

            #         with torch.no_grad():
            #             if self.clip_cls_style == "cropmask":
            #                 clip_input_images = box_masked_image_tensor
            #             elif self.clip_cls_style == "mask":
            #                 clip_input_images = masked_image_tensor
            #             elif self.clip_cls_style == "crop":
            #                 clip_input_images = box_image_tensor
            #             else:
            #                 raise NotImplementedError

            #             image_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_input_images)
            #             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #             logit_scale = self.sem_seg_head.predictor.clip_model.logit_scale.exp()
            #             logits_per_image = logit_scale.half() * image_features @ self.sem_seg_head.predictor.text_features_test_clip.t().half()
            #             logits_per_image = logits_per_image.float()
            #             logits_per_image = torch.cat((logits_per_image, mask_cls_result[:, -1].unsqueeze(1)), 1)
            #             assert not (self.ensembling and self.ensembling_all_cls)
            #             if self.ensembling:
            #                 # note that in this branch, average the seen score of clip
            #                 # seen_indexes, unseen_indexes = self.seen_unseen_indexes()
            #                 lambda_balance = 2 / 3.
            #                 mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
            #                 # shape of logits_per_image: [100, 171]
            #                 logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
            #                 # remove the influence of clip on seen classes
            #                 logits_per_image[:, self.seen_indexes] = logits_per_image[:, self.seen_indexes].mean(dim=1, keepdim=True)

            #                 mask_cls_result[:, self.seen_indexes] = torch.pow(mask_cls_result[:, self.seen_indexes], lambda_balance) \
            #                                                    * torch.pow(logits_per_image[:, self.seen_indexes], 1 - lambda_balance)
            #                 mask_cls_result[:, self.unseen_indexes] = torch.pow(mask_cls_result[:, self.unseen_indexes], 1 - lambda_balance) \
            #                                                    * torch.pow(logits_per_image[:, self.unseen_indexes], lambda_balance)
            #             elif self.ensembling_all_cls:
            #                 lambda_balance = 2 / 3.
            #                 mask_cls_result = F.softmax(mask_cls_result, dim=-1)[..., :-1]
            #                 logits_per_image = F.softmax(logits_per_image, dim=-1)[..., :-1]
            #                 mask_cls_result = torch.pow(mask_cls_result, 1 - lambda_balance) \
            #                                                    * torch.pow(logits_per_image, lambda_balance)
            #             else:
            #                 mask_cls_result = logits_per_image

            #         ######################################################################################
            #     if self.sem_seg_postprocess_before_inference:
            #         mask_pred_result = sem_seg_postprocess(
            #             mask_pred_result, image_size, height, width
            #         )

            #     # semantic segmentation inference
            #     if (self.clip_classification and self.ensembling) or (self.clip_classification and self.ensembling_all_cls):
            #         r = self.semantic_inference2(mask_cls_result, mask_pred_result)
            #     else:
            #         r = self.semantic_inference(mask_cls_result, mask_pred_result)
            #     if not self.sem_seg_postprocess_before_inference:
            #         r = sem_seg_postprocess(r, image_size, height, width)
            #     #############################################################################
            #     # gzero calibrate
            #     if self.gzero_calibrate > 0:
            #         r[self.seen_indexes, :, :] = r[self.seen_indexes, :, :] - self.gzero_calibrate
            #     ###########################################################################
            #     processed_results.append({"sem_seg": r})

            # return processed_results
            
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # if self.clip_classification:
            #     m
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        # print('o ',output.shape)
        # torch.Size([1, 8, 124, 120, 120])
        
        return prediction
        # return output
   
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
