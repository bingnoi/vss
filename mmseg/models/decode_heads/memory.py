import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        # self.memory = nn.Parameter(torch.zeros([1,5,512,15,15]), requires_grad = False)
        self.memory = nn.Parameter(torch.zeros([1,512,15,15]), requires_grad = False)
        self.linearfuse1 = nn.Linear(5*512,512)
        
        self.linear1 = nn.Linear(512,512)
        self.linear2 = nn.Linear(512,512)
        
    def set_memory(self,feats):
        self.memory.data=feats
        return self.memory.data

    def init_memory(self,feats):
        #print(feats.shape)
        b,n,c,hx,wx = feats.shape
        feats = feats.reshape(b,-1,hx,wx).permute(0,2,3,1)
        feats = self.linearfuse1(feats)
        feats = feats.permute(0,3,1,2) 
        
        self.memory.data = feats
        return self.memory.data

    def update_memory(self,feats):
        B,num_clips,cx,hx,wx=feats.shape
        
        b,n,c,hx,wx = feats.shape
        feats = feats.reshape(b,-1,hx,wx).permute(0,2,3,1)
        feats = self.linearfuse1(feats)
        feats = feats.permute(0,3,1,2)#b,c,hw,wx
        
        query_frame_selected = feats.permute(0,2,3,1).reshape(B,-1,cx)
        query_frame_selected = self.linear1(query_frame_selected)
        
        # print('stegsgbbbbbbbbb',self.memory.data.shape)
        memory_f = self.memory.data.permute(0,2,3,1).reshape(B,-1,cx)
        memory_feature = self.linear2(memory_f)

        # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
        atten = torch.matmul(memory_feature,query_frame_selected.transpose(-1,-2))

        #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
        out = torch.matmul(atten,query_frame_selected).reshape(B,hx,wx,cx).permute(0,3,1,2)
        self.memory.data = out
        return self.memory.data

    def forward(self,mode,feats):
        # for u in feats:
        #     print(u.shape)
        feats = feats.unsqueeze(0)
        
        if mode == 'init_memory':
            return self.init_memory(feats)
        elif mode == 'update_memory':
            return self.update_memory(feats)
        elif mode == 'set_memory':
            return self.set_memory(feats)