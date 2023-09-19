import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .transformer.position_encoding import PositionEmbeddingSine
from .transformer.transformer import Transformer
from .third_party import clip
from .third_party import imagenet_templates

import numpy as np

class  TransformerZeroshotPredictor(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, x, mask_features, images_tensor=None, ori_sizes=None):
        assert images_tensor == None

        pos = self.pe_layer(x)
        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            x_cls = self.projection_layer(hs)
            # TODO: check if it is l2 norm
            x_cls = x_cls / x_cls.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            if self.training:
                cls_score = logit_scale * x_cls @ self.text_features.clone().detach().t()
            else:
                cls_score = logit_scale * x_cls @ self.text_features_test.clone().detach().t()

            bg_score = logit_scale * x_cls @ self.bg_feature.t()
            outputs_class = torch.cat((cls_score, bg_score), -1)
            out = {"pred_logits": outputs_class[-1]}

        else:
            out = {}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks

        return out
        
        