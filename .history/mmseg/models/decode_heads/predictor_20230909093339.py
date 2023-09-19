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
        
        hidden_dim = 256
        dropout = 0.1
        nheads = 8
        dim_feedforward = 2048
        enc_layers = 0
        dec_layers = 6
        pre_norm = False
        deep_supervision = True
        
        #not certain
        num_queries = 256
        in_channels = 256
        
        self.mask_classification = True
        
        if self.wordvec:
            self.projection_layer = nn.Linear(hidden_dim, 600)
        else:
            self.projection_layer = nn.Linear(hidden_dim, 512)
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
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
        
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x