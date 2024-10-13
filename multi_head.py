import torch
import torch.nn as nn

class Multi_head_self_attention(nn.Module):
    def __init__(self,dim_q,dim_k,dim_v):

        self.n_heads = n_heads
        
        self.query_emb = nn.Linear(dim_q,dim_q)
        self.key_emb = nn.Linear(dim_k,dim_k)
        self.value_emb = nn.Linear(dim_v,dim_v)

        self.n_softmax = nn.Softmax()

        self.dim_k = dim_k
    
    def forward(self,x,y):
        query = self.query_emb(x).reshape()
        key = self.key_emb(y)
        value = self.value_emb(y)

        attention_map = torch.matmul(query,key.transpose(-1,-2))
        attention_map = attention_map/self.dim_k
        attention_map = self.n_softmax(attention_map)
        final = torch.matmul(attention_map,value) 
        return final

x = torch.tensor([1,3,50,50])
y = torch.tensor([1,3,80,80])
attention_fun = Multi_head_self_attention(50,50,80)
attention_fun.forward(x,y)
