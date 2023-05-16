import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        self.memory = nn.Parameter(torch.zeros([1,3,512,15,15]), requires_grad = False)

    def init_memory(self,feats):
        self.memory = torch.cat(feats,dim=2)
        return self.memory

    def update_memory(self,feats):
        B,num_clips,cx,hx,wx=feats.shape
        query_frame_selected = feats.permute(0,1,3,4,2).reshape(B,num_clips,-1,cx)

        query_frame_selected = self.linear1(query_frame_selected)
        memory_feature = self.linear2(self.memory.data.permute(0,1,3,4,2)).reshape(B,num_clips,-1,cx)

        # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
        atten = torch.matmul(memory_feature,query_frame_selected.transpose(-1,-2))

        #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
        out = torch.matmul(atten,query_frame_selected).reshape(B,num_clips,hx,wx,cx).permute(0,1,4,2,3)
        return out

    def forward(self,mode,feats):
        print('f ',feats.shape)
        if mode == 'init_memory':
            self.init_memory(feats)
        elif mode == 'update_memory':
            self.update_memory(feats)
        return self.memory

    