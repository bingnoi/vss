import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        # self.memory = nn.Parameter(torch.zeros([1,5,512,15,15]), requires_grad = False)
        self.memory = nn.Parameter(torch.zeros([1,512,15,15]), requires_grad = False)
        
        self.linear1 = nn.Linear(512,512)
        self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(512,512)
        
        self.initlinear = nn.Linear(512,512)
        self.initlinear2 = nn.Linear(512,512)
        self.initlinear3 = nn.Linear(512,512)
        
        # self.down_conv = nn.Conv2d(512 * 4, 512 , kernel_size=3, stride=1, padding=1, bias=False)
        
        # self.down_linear = nn.Linear(4*512,512)
        
    def set_memory(self,feats):
        self.memory.data=feats
        return self.memory.data
    
    def handle_squeeze(self,feats):
        b,n,cx,hx,wx = feats.shape
        
        # feats = feats.reshape(b,n*cx,hx,wx)
        
        # feats = self.down_linear(feats.reshape(b,n*cx,hx,wx).permute(0,2,3,1)).permute(0,3,1,2)
        
        f_first = feats[:,:1]
        f_later = feats
        
        f_first = f_first.permute(0,1,3,4,2).reshape(b,1*hx*wx,cx)
        f_later = f_later.permute(0,1,3,4,2).reshape(b,n*hx*wx,cx)
        
        # print('f',f_later.shape)
        f_first_q = self.initlinear(f_first)
        f_later_k = self.initlinear2(f_later)
        f_later_v = self.initlinear3(f_later)
        
        
        feats_atten = torch.matmul(f_first_q,f_later_k.transpose(-1,-2)) #b,1*hx*wx,cx b,cx,n*hx*wx
        feats = torch.matmul(feats_atten,f_later_v) #b,hx*wx,n*hx*wx b,n*hx*wx,cx = b,hx*wx,cx
        
        # feats = feats.permute(0,2,1,3).reshape(b,hx*wx,n*cx)
        feats = feats.reshape(b,hx,wx,cx).permute(0,3,1,2)
        
        # feats = self.down_conv(feats)
        
        return feats
        
    def init_memory(self,feats):
        #print(feats.shape)
        feats = self.handle_squeeze(feats)
        
        self.memory.data = feats
        return self.memory.data

    def update_memory(self,feats):
        B,n,cx,hx,wx = feats.shape
        
        feats = self.handle_squeeze(feats)
        feats = feats.reshape(B,cx,hx*wx).permute(0,2,1)
        
        # print('f',feats.shape) #torch.Size([1, 512, 15, 27])
        # exit()
        feats_k = self.linear1(feats)
        feats_v = self.linear2(feats)
        
        # print('stegsgbbbbbbbbb',self.memory.data.shape)
        b_m,c_m,hx_m,wx_m = self.memory.data.shape
        memory_f = self.memory.data.permute(0,2,3,1).reshape(b_m,-1,c_m)
        memory_feature = self.linear3(memory_f)

        # torch.Size([1, 3, 225, 512]) torch.Size([1, 3, 512, 225])
        
        atten = torch.matmul(memory_feature,feats_k.transpose(-1,-2)) #b,h*w,h*w b,n,h*w,c = b,n,h*w,c

        #[1,3,225,225]*[1,3,225,512] = [1,3,225,512]
        # print('shape',torch.matmul(atten,feats_v).shape,b_m,hx_m,wx_m,c_m)
        # exit()
        out = torch.matmul(atten,feats_v).permute(0,2,1).reshape(B,cx,hx,wx)
        
        # out = self.down_conv(out)
        
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