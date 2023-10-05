import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

from scipy.optimize import linear_sum_assignment

#multi-scale fusion
#multi-frame fusion
#correlation refine

# class SelfAttentionLayer(nn.Module):

#     def __init__(self, d_model, nhead, dropout=0.0,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt,
#                      tgt_mask: Optional[Tensor] = None,
#                      tgt_key_padding_mask: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None):
#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)

#         return tgt

#     def forward_pre(self, tgt,
#                     tgt_mask: Optional[Tensor] = None,
#                     tgt_key_padding_mask: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
#         tgt2 = self.norm(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout(tgt2)
        
#         return tgt

#     def forward(self, tgt,
#                 tgt_mask: Optional[Tensor] = None,
#                 tgt_key_padding_mask: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(tgt, tgt_mask,
#                                     tgt_key_padding_mask, query_pos)
#         return self.forward_post(tgt, tgt_mask,
#                                  tgt_key_padding_mask, query_pos)


# class CrossAttentionLayer(nn.Module):

#     def __init__(self, d_model, nhead, dropout=0.0,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt, memory,
#                      memory_mask: Optional[Tensor] = None,
#                      memory_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None,
#                      query_pos: Optional[Tensor] = None):
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)
        
#         return tgt

#     def forward_pre(self, tgt, memory,
#                     memory_mask: Optional[Tensor] = None,
#                     memory_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None,
#                     query_pos: Optional[Tensor] = None):
#         tgt2 = self.norm(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory, attn_mask=memory_mask,
#                                    key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout(tgt2)

#         return tgt

#     def forward(self, tgt, memory,
#                 memory_mask: Optional[Tensor] = None,
#                 memory_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None,
#                 query_pos: Optional[Tensor] = None):
#         if self.normalize_before:
#             return self.forward_pre(tgt, memory, memory_mask,
#                                     memory_key_padding_mask, pos, query_pos)
#         return self.forward_post(tgt, memory, memory_mask,
#                                  memory_key_padding_mask, pos, query_pos)


# class FFNLayer(nn.Module):

#     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm = nn.LayerNorm(d_model)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#         self._reset_parameters()
    
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos

#     def forward_post(self, tgt):
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)
#         return tgt

#     def forward_pre(self, tgt):
#         tgt2 = self.norm(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout(tgt2)
#         return tgt

#     def forward(self, tgt):
#         if self.normalize_before:
#             return self.forward_pre(tgt)
#         return self.forward_post(tgt)

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# class Multi_Decoder(nn.Module):
#     def __init__(self,
#                  hidden_dim=512,
#                  nheads=8,
#                  pre_norm=False,
#                  dim_feedforward=2048,
#                  num_queries=100) -> None:
#         super().__init__()
#         N_steps = hidden_dim//2
        
#         self.num_feature_levels = 3
#         self.num_layers = 7
#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
        
#         self.num_queries = num_queries
#         # learnable query features
#         self.query_feat = nn.Embedding(num_queries, hidden_dim)
#         # learnable query p.e.
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
            
#     def forward(self,x,per_pixel_feature):
#         assert len(x) == self.num_feature_levels
#         src = []
#         pos = []
#         size_list = []
        
#         for i in range(self.num_feature_levels):
#             size_list.append(x[i].shape[-2:])
#             pos.append(self.pe_layer(x[i], None).flatten(2))
#             src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
        
#             pos[-1] = pos[-1].permute(2, 0, 1)
#             src[-1] = src[-1].permute(2, 0, 1)
        
#         _, bs, _ = src[0].shape
        
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
#         output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
#             output = self.transformer_cross_attention_layers[i](
#                 output, src[level_index],
#                 memory_mask=attn_mask,
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=pos[level_index], query_pos=query_embed
#             )

#             output = self.transformer_self_attention_layers[i](
#                 output, tgt_mask=None,
#                 tgt_key_padding_mask=None,
#                 query_pos=query_embed
#             )
            
#             # FFN
#             output = self.transformer_ffn_layers[i](
#                 output
#             )

#             outputs_object, outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
#             predictions_object.append(outputs_object)
#             predictions_class.append(outputs_class)
#             predictions_mask.append(outputs_mask)

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # if guidance is not None:
        #     T = x.size(0) // guidance.size(0)
        #     guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
        #     x = torch.cat([x, guidance], dim=1)
        return self.conv(x)

class Corr(nn.Module):
    def __init__(self,
                 prompt_channel = 8,
                 hidden_dim = 128,
                 num_feature_scale = 4
                 ) -> None:
        super().__init__()
        decoder_guidance_dims = [64,128,320,512]
        decoder_guidance_proj_dims = [512] * 4
        
        decoder_dims = [64,128,320,512]
        
        self.num_feature_scale = num_feature_scale
        self.num_frames = 4
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
    
        self.decoder1 = Up(hidden_dim, decoder_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1])
        self.decoder3 = Up(decoder_dims[1], decoder_dims[2])
        self.decoder4 = Up(decoder_dims[2], decoder_dims[3])
        self.head = nn.Conv2d(decoder_dims[3], 1, kernel_size=3, stride=1, padding=1)
    
    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr
    
    def corr_embed(self, x):
        B = x.shape[0]
        print('x',x.shape)
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def conv_decoder(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        print('q',corr_embed.shape)
        corr_embed = self.decoder1(corr_embed)
        corr_embed = self.decoder2(corr_embed)
        corr_embed = self.decoder3(corr_embed)
        corr_embed = self.decoder4(corr_embed)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self,img,text):
        correlation_c1 = []
        correlation_c2 = []
        correlation_c3 = []
        correlation_c4 = []
        
        for i in range(self.num_feature_scale):
            for j in range(self.num_frames):
                if i==0:
                    correlation_c1.append(self.correlation(img[i][j],text))
                elif i==1:
                    correlation_c2.append(self.correlation(img[i][j],text))
                elif i==2:
                    correlation_c3.append(self.correlation(img[i][j],text))
                elif i==3:
                    correlation_c4.append(self.correlation(img[i][j],text))
        
        print([i.shape for i in correlation_c1])
        exit()
        # corr_embed = self.corr_embed(corr)
        
        # print('s0',img.shape,text.shape)
        # print('s1',corr.shape,corr_embed.shape)
        
        # s0 torch.Size([4, 512, 24, 24]) torch.Size([4, 111, 8, 512])
        # s1 torch.Size([4, 8, 111, 24, 24]) torch.Size([4, 512, 111, 24, 24])
        
        logit = self.conv_decoder(corr_embed)

        print(img.shape,text.shape,corr.shape,corr_embed.shape,logit.shape)

        return logit