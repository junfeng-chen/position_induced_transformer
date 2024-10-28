import torch
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
import torch.nn as nn
from torch.nn.functional import gelu
import numpy as np
np.random.seed(0)
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

class posatt(nn.Module):
    def __init__(self, n_head, in_dim, locality):
        super(posatt, self).__init__()        
        
        self.locality  = locality
        self.n_head    = n_head
        self.in_dim    = in_dim
        self.lmda      = torch.nn.Parameter( torch.rand(n_head, 1, 1) ) 

    def forward(self, mesh, inputs):
        """
        mesh: (batch, L, 2)
        inputs: (batch, L_in, in_dim)
        """
        att        = self.dist2att(mesh, mesh, self.lmda, self.locality)  # (n_head, L, L)
        convoluted = self.convolution(att, inputs)
        return torch.cat((inputs, convoluted), dim=-1)

    def dist2att(self, mesh_out, mesh_in, scale, locality):
        m_dist      = torch.sum((mesh_out.unsqueeze(-2) - mesh_in.unsqueeze(-3))**2, dim=-1) # (batch_size, L_out, L_in)
        scaled_dist = m_dist.unsqueeze(1) * torch.tan(0.25*pi*(1-1e-7)*(1.0+torch.sin(scale))) # (batch_size, n_head, L_out, L_in)
        mask        = torch.quantile(scaled_dist, locality, dim=-1, keepdim=True) 
        scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device))
        scaled_dist = -scaled_dist
        return torch.nn.Softmax(dim=-1)(scaled_dist)

    def convolution(self, A, U):
        convoluted = torch.einsum("bhnj,bjd->bnhd", A, U)  # (batch, L_out, n_head * in_dim) 
        convoluted = convoluted.reshape(U.shape[0], -1, self.n_head * U.shape[-1])  # (batch, L_out, n_head*hid_dim)
        return convoluted

class posatt_cross(posatt):
    def __init__(self, n_head, in_dim, locality):
        super(posatt_cross, self).__init__(n_head, in_dim, locality)

    def forward(self, mesh_out, mesh_in, inputs):
        """
        mesh_out: (batch, L_out, 2)
        mesh_in: (batch, L_in, 2)
        inputs: (batch, L_in, in_dim)
        """
        att        = self.dist2att(mesh_out, mesh_in, self.lmda, self.locality)
        convoluted = self.convolution(att, inputs)
        return convoluted

class pit(nn.Module):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):

        super(pit, self).__init__()
        self.space_dim= space_dim
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.hid_dim  = hid_dim
        self.n_head   = n_head
        self.n_blocks = n_blocks
        if mesh_ltt != None:
            self.mesh_ltt = mesh_ltt.reshape(-1, self.space_dim)
        else:
            self.mesh_ltt = mesh_ltt
        self.en_local = en_loc
        self.de_local = de_loc

        self.down     = posatt_cross(self.n_head, self.in_dim, self.en_local)
        self.en_layer = kaiming_mlp(self.n_head * (self.in_dim+self.space_dim), self.hid_dim, self.hid_dim)
        
        self.conv     = torch.nn.ModuleList([posatt(self.n_head, self.hid_dim, 1.0) for _ in range(self.n_blocks)])
        self.mlp      = torch.nn.ModuleList([kaiming_mlp((1 + self.n_head) * self.hid_dim, self.hid_dim, self.hid_dim) for _ in range(self.n_blocks)]) # with residual in the convolutions

        self.up       = posatt_cross(self.n_head, self.hid_dim, self.de_local)
        self.de       = kaiming_mlp(self.n_head * self.hid_dim, self.hid_dim, self.out_dim)

    def encoder(self, mesh_in, func_in, mesh_ltt):
        func_ltt = self.down(mesh_ltt, mesh_in, func_in)
        func_ltt = self.en_layer(func_ltt)
        func_ltt = gelu(func_ltt )
        return func_ltt 

    def processor(self, func_ltt, mesh_ltt):
        for a, w in zip(self.conv, self.mlp):
            '''
            U = AUW
            '''
            func_ltt  = a(mesh_ltt, func_ltt)
            func_ltt  = w(func_ltt)
            func_ltt  = gelu(func_ltt)
        return func_ltt 

    def decoder(self, mesh_ltt, func_ltt, mesh_out):
        func_out = self.up(mesh_out, mesh_ltt, func_ltt) 
        func_out = self.de(func_out)
        return func_out

class posatt_fixed(posatt):
    def __init__(self, n_head, in_dim, locality):
        super(posatt_fixed, self).__init__(n_head, in_dim, locality)

    def dist2att(self, mesh_out, mesh_in, scale, locality):
        m_dist      = torch.sum((mesh_out.unsqueeze(-2) - mesh_in.unsqueeze(-3))**2, dim=-1) # (L_out, L_in)
        scaled_dist = m_dist * torch.tan(0.25*pi*(1-1e-7)*(1.0+torch.sin(scale))) # (n_head, L_out, L_in)
        mask        = torch.quantile(scaled_dist, locality, dim=-1, keepdim=True) 
        scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device))
        scaled_dist = -scaled_dist
        return torch.nn.Softmax(dim=-1)(scaled_dist)

    def convolution(self, A, U):
        convoluted = torch.einsum("hnj,bjd->bnhd", A, U)  # (batch, L_out, n_head * in_dim) 
        convoluted = convoluted.reshape(U.shape[0], -1, self.n_head * U.shape[-1])  # (batch, L_out, n_head*hid_dim)
        return convoluted

class posatt_cross_fixed(posatt_fixed):

    def __init__(self, n_head, in_dim, locality):
        super(posatt_cross_fixed, self).__init__(n_head, in_dim, locality)

    def forward(self, mesh_out, mesh_in, inputs):
        """
        mesh_out: (L_out, 2)
        mesh_in: (L_in, 2)
        inputs: (batch, L_in, in_dim)
        """
        att        = self.dist2att(mesh_out, mesh_in, self.lmda, self.locality)
        convoluted = self.convolution(att, inputs)
        return convoluted

class pit_fixed(pit):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):

        super(pit_fixed, self).__init__(space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc)
        self.down     = posatt_cross_fixed(self.n_head, self.in_dim, self.en_local)
        self.conv     = torch.nn.ModuleList([posatt_fixed(self.n_head, self.hid_dim, 1.0) for _ in range(self.n_blocks)])
        self.up       = posatt_cross_fixed(self.n_head, self.hid_dim, self.de_local)
##############################
class posatt_periodic1d(posatt_fixed):
    def __init__(self, n_head, in_dim, locality):
        super(posatt_periodic1d, self).__init__(n_head, in_dim, locality)

    def dist2att(self, mesh_out, mesh_in, scale, locality):
        dx          = torch.abs(mesh_in[1,0] - mesh_in[0,0])
        l           = dx * mesh_in.shape[0]
        m_diff      = abs(mesh_out.unsqueeze(-2) - mesh_in.unsqueeze(-3))
        m_diff      = torch.minimum(m_diff, l-m_diff)
        m_dist      = m_diff[...,0]**2
        scaled_dist = m_dist * torch.tan(0.25*pi*(1-1e-7)*(1.0+torch.sin(scale))) # (n_head, L_out, L_in)
        mask        = torch.quantile(scaled_dist, locality, dim=-1, keepdim=True) 
        scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device))
        scaled_dist = -scaled_dist
        return torch.nn.Softmax(dim=-1)(scaled_dist)

class posatt_cross_periodic1d(posatt_periodic1d):

    def __init__(self, n_head, in_dim, locality):
        super(posatt_cross_periodic1d, self).__init__(n_head, in_dim, locality)

    def forward(self, mesh_out, mesh_in, inputs):
        """
        mesh_out: (L_out, 2)
        mesh_in: (L_in, 2)
        inputs: (batch, L_in, in_dim)
        """
        att        = self.dist2att(mesh_out, mesh_in, self.lmda, self.locality)
        convoluted = self.convolution(att, inputs)
        return convoluted

class pit_periodic1d(pit):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):

        super(pit_periodic1d, self).__init__(space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc)
        self.down     = posatt_cross_periodic1d(self.n_head, self.in_dim, self.en_local)
        self.conv     = torch.nn.ModuleList([posatt_periodic1d(self.n_head, self.hid_dim, 1.0) for _ in range(self.n_blocks)])
        self.up       = posatt_cross_periodic1d(self.n_head, self.hid_dim, self.de_local)
################

class posatt_periodic2d(posatt_fixed):
    def __init__(self, n_head, in_dim, locality):
        super(posatt_periodic2d, self).__init__(n_head, in_dim, locality)

    def dist2att(self, mesh_out, mesh_in, scale, locality):
        res         = int(mesh_in.shape[0]**0.5)
        dx          =( torch.max(mesh_in[:,0]) - torch.min(mesh_in[:,0]) ) / (res - 1)
        l           = dx * res
        m_diff      = abs(mesh_out.unsqueeze(-2) - mesh_in.unsqueeze(-3))
        m_diff      = torch.minimum(m_diff, l-m_diff)
        m_dist      = torch.sum(m_diff**2, dim=-1)
        scaled_dist = m_dist * torch.tan(0.25*pi*(1-1e-7)*(1.0+torch.sin(scale))) # (n_head, L_out, L_in)
        mask        = torch.quantile(scaled_dist, locality, dim=-1, keepdim=True) 
        scaled_dist = torch.where(scaled_dist <= mask, scaled_dist, torch.tensor(torch.finfo(torch.float32).max, device=scaled_dist.device))
        scaled_dist = -scaled_dist
        return torch.nn.Softmax(dim=-1)(scaled_dist)

class posatt_cross_periodic2d(posatt_periodic2d):

    def __init__(self, n_head, in_dim, locality):
        super(posatt_cross_periodic2d, self).__init__(n_head, in_dim, locality)

    def forward(self, mesh_out, mesh_in, inputs):
        """
        mesh_out: (L_out, 2)
        mesh_in: (L_in, 2)
        inputs: (batch, L_in, in_dim)
        """
        att        = self.dist2att(mesh_out, mesh_in, self.lmda, self.locality)
        convoluted = self.convolution(att, inputs)
        return convoluted

class pit_periodic2d(pit):
    def __init__(self,
                 space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc):

        super(pit_periodic2d, self).__init__(space_dim,  
                 in_dim, 
                 out_dim, 
                 hid_dim,
                 n_head, 
                 n_blocks,
                 mesh_ltt,
                 en_loc, 
                 de_loc)
        self.down     = posatt_cross_periodic2d(self.n_head, self.in_dim, self.en_local)
        self.conv     = torch.nn.ModuleList([posatt_periodic2d(self.n_head, self.hid_dim, 1.0) for _ in range(self.n_blocks)])
        self.up       = posatt_cross_periodic2d(self.n_head, self.hid_dim, self.de_local)
