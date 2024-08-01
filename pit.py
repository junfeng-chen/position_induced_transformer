import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import gelu
# from pytorch3d.ops import sample_farthest_points as batch_fps
from math import pi


class kaiming_mlp(nn.Module): 
    def __init__(self, n_filters0, n_filters1, n_filters2): 
        super(kaiming_mlp, self).__init__() 
        self.mlp1 = torch.nn.Linear(n_filters0, n_filters1)
        self.mlp2 = torch.nn.Linear(n_filters1, n_filters2)
        nn.init.kaiming_normal_(self.mlp1.weight)
        nn.init.kaiming_normal_(self.mlp2.weight)
 
    def forward(self, x): 
        x = self.mlp1(x)
        x = gelu(x)
        x = self.mlp2(x)
        return x

class mhpa(nn.Module):

    def __init__(self, n_head, hid_dim, locality):

        super(mhpa, self).__init__()        
        
        self.locality = locality 
        self.hid_dim = hid_dim 
        self.n_head = n_head 
        self.v_dim = round(hid_dim / n_head) 
 
        self.r = torch.nn.Parameter( torch.rand(1, n_head, 1, 1) ) 
        self.weight = torch.nn.Parameter( torch.rand(n_head, hid_dim, self.v_dim) )
        torch.nn.init.kaiming_normal_(self.weight) 
 
    def forward(self, m_dist, inputs):
        scaled_dist = m_dist.unsqueeze(1) * torch.tan(0.25*pi*(1.0+torch.sin(self.r)))
        if self.locality <= 1: 
            mask = torch.quantile(scaled_dist, self.locality, dim=-1, keepdim=True) 
            scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device)) 
         
        scaled_dist = -scaled_dist
        att = torch.nn.Softmax(dim=-1)(scaled_dist)  # (batch, n_heads, N, N) 
 
        value = torch.einsum("bnj,hjk->bhnk", inputs, self.weight)  # (batch, n_head, N, v_dim) 
 
        concat = torch.einsum("bhnj,bhjd->bhnd", att, value)  # (batch, n_head, N, v_dim) 
        concat = concat.permute(0, 2, 1, 3).contiguous() 
        concat = concat.view(inputs.shape[0], -1, self.hid_dim) 
 
        return gelu(concat)

class pit(nn.Module):

    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks, 
                 l_latent, 
                 en_loc, 
                 de_loc):

        super(pit, self).__init__()
        self.space_dim= space_dim
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.n_blocks = n_blocks
        self.l_ltt    = l_latent
        self.en_local = en_loc
        self.de_local = de_loc

        self.en_layer = torch.nn.Linear(self.in_dim, self.hid_dim)
        torch.nn.init.kaiming_normal_(self.en_layer.weight) 
        self.down     = mhpa(self.n_head, self.hid_dim, self.en_local)
        
        
        self.PA       = torch.nn.ModuleList([mhpa(self.n_head, self.hid_dim, 2) for _ in range(self.n_blocks)]) 
        self.MLP      = torch.nn.ModuleList([kaiming_mlp(self.hid_dim, self.hid_dim, self.hid_dim) for _ in range(self.n_blocks)])
        self.W        = torch.nn.ModuleList([torch.nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_blocks)])
        for linear in self.W: 
            torch.nn.init.kaiming_normal_(linear.weight) 
        
        self.up = mhpa(self.n_head, self.hid_dim, self.de_local)
        self.de_layer = kaiming_mlp(self.hid_dim, self.hid_dim, self.out_dim)
 
    def forward(self, x):

        original_size = x.shape[:-1]
        x, m_in, m_ltt = self.pairwise_dist(x)

        x    = gelu(self.en_layer(x)) 
        x    = self.down(m_in, x) 
 
        for pa, mlp, w in zip(self.PA, self.MLP, self.W): 
            x = mlp(pa(m_ltt, x)) + w(x) 
            x = gelu(x) 
 
        x = self.up(m_in.permute(0,2,1), x) 
        x = self.de_layer(x) 
        return x.reshape(*original_size, self.out_dim)
    
    def pairwise_dist(self, x):
        # ltt, _ = batch_fps(ext, K=self.l_ltt, random_start_point=False)
        ltt    = x[:, ::2, ::2, :][:, :111, :26, :].reshape(x.shape[0], -1, self.space_dim)
        x    = x.reshape(x.shape[0], -1, self.space_dim)
        m_in   = torch.cdist(ltt, x, p=2.0, compute_mode='use_mm_for_euclid_dist')**2
        m_ltt  = torch.cdist(ltt, ltt, p=2.0, compute_mode='use_mm_for_euclid_dist')**2
        return x, m_in, m_ltt
