import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .transformer.position_encoding import PositionEmbeddingSine
from .transformer.transformer import Transformer
from .third_party import clip
from .third_party import imagenet_templates

from einops import rearrange

from .transformer.models import  Aggregator

import numpy as np

# from .hyper_correlation import Corr
from .hyper_co import Corr

class  CatClassifier(nn.Module):
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
        self.mask_classification = True
        self.wordvec = False
        train_class_json = "/home/lixinhao/vss/mmseg/handle_data/seen_classnames.json"
        test_class_json = "/home/lixinhao/vss/mmseg/handle_data/unseen_classnames.json"
        clip_pretrained = "ViT-B/16"
        # clip_pretrained = "ViT-L/14@336px"
        prompt_ensemble_type = "imagenet"
        train_class_indexes_json = "datasets/coco/coco_stuff/split/seen_indexes.json"
        test_class_indexes_json = "datasets/coco/coco_stuff/split/seen_indexes.json"
        clip_classification = True
        mask_classification =True
        
        #not certain
        num_queries = 100
        in_channels = 256
        enforce_input_project = False
        temperature = 0.01
        
        text_dim=512
        text_guidance_proj_dim = 128
        appearance_guidance_dim =256
        appearance_guidance_proj_dim = 128
        decoder_dims = [64,32]
        decoder_guidance_dims = [256,128]
        decoder_guidance_proj_dims = [32,16]
        num_layers = 2
        num_heads = 4
        hidden_dims =128
        pooling_sizes = [2,2]
        feature_resolution = [24,24]
        window_sizes = 12
        attention_type = "linear"
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
        
        self.prompt_ensemble_type = prompt_ensemble_type
        
        assert "A photo of" not in self.class_texts[0]
        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        prompt_templates_clip = imagenet_templates.IMAGENET_TEMPLATES_SELECT_CLIP
        
        self.clip_classification = clip_classification
        if self.clip_classification:
            self.clip_model = clip_model.float()
            self.clip_preprocess = clip_preprocess
        
        
        clip_finetune = "attention"
        for name, params in self.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        # finetune_backbone = 0.01 > 0.
        # for name, params in self.backbone.named_parameters():
        #     if "norm0" in name:
        #         params.requires_grad = False
        #     else:
        #         params.requires_grad = finetune_backbone
        
        ## train features
        # shape of text_features: [156, 512]
        self.text_features = zeroshot_classifier(self.class_texts, prompt_templates, clip_model).permute(1, 0,2).float()
        self.text_features_test = zeroshot_classifier(self.test_class_texts, prompt_templates, clip_model).permute(1, 0,2).float()

        transformer = Aggregator(
            text_guidance_dim=text_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=8
            )
        self.transformer = transformer
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        
        self.corr = Corr()
        
    def forward(self, ori_images,fuse_f,c4, images_tensor=None, ori_sizes=None):
        assert images_tensor == None
        
        ori_images = torch.cat([ori_images],dim=0)
        ori_images = F.interpolate(ori_images, size=self.clip_resolution, mode='bilinear', align_corners=False)
        clip_features = self.clip_model.encode_image(ori_images, dense=True)

        feature_resolution = [24,24]
        img_feat = rearrange(clip_features[:, 1:, :], "b (h w) c->b c h w", h=feature_resolution[0], w=feature_resolution[1])
        
        # b,c,h,w = fuse_f[0].shape
        # for idx,i in enumerate(fuse_f[1:]):
        #     resize_i = i.reshape(b,h,w,c).permute(0,3,1,2)
        #     fuse_f[idx+1] = F.interpolate(resize_i,size=self.clip_resolution, mode='bilinear', align_corners=False) 
        # vis = fuse_f[::-1]
        
        text = self.text_features if self.training else self.text_features_test
        # print('text1',text.shape)
        text = text.repeat(img_feat.shape[0], 1, 1, 1)
        # print('text2',text.shape)
        
        # out = self.transformer(img_feat, text, vis)
        out = self.corr(fuse_f,img_feat,c4,text)
        
        return out
    
def zeroshot_classifier(classnames, templates, clip_modelp):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize, shape: [48, 77]
            class_embeddings = clip_modelp.encode_text(texts)  # embed with text encoder
            # print('p',texts.shape,class_embeddings.shape)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # class_embedding = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
            class_embeddings /= class_embeddings.norm()
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
        