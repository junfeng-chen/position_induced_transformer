import operator
from functools import reduce
import torch
import torch.nn.functional as F

class PixelWiseNormalization():
    def __init__(self, x, eps=1e-5):
        
        self.mean = torch.mean(x, dim=0, keepdim=True)  # (1, h, w, 1)
        self.std = torch.std(x, dim=0, keepdim=True) #(1,h,w,1)
        self.eps = eps
        
    def normalize(self, x):
        try:
            x = (x - self.mean) / (self.std + self.eps)
        except:#do upsampling 
            h = x.shape[1]
            w = x.shape[2]
            mean = F.interpolate(self.mean.permute(0,3,1,2), size=(h, w), mode='bilinear', align_corners=False).permute(0,2,3,1)
            std  = F.interpolate(self.std.permute(0,3,1,2), size=(h, w), mode='bilinear', align_corners=False).permute(0,2,3,1)
            x = (x - mean) / (std + self.eps)
        return x

    def denormalize(self, x):
        
        try:
            x    = x * (self.std + self.eps) + self.mean
        except:#do upsampling 
            h = x.shape[1]
            w = x.shape[2]
            mean = F.interpolate(self.mean.permute(0,3,1,2), size=(h, w), mode='bilinear', align_corners=False).permute(0,2,3,1)
            std  = F.interpolate(self.std.permute(0,3,1,2), size=(h, w), mode='bilinear', align_corners=False).permute(0,2,3,1)
            x    = x * (std + self.eps) + mean
        return x

    def to(self, device):
        if device == 'cpu':
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
        elif device == 'cuda':
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()))
    return c

class RelMaxNorm(object):
    def __init__(self, out_dim):
        super(RelMaxNorm, self).__init__()
        self._out_dim = out_dim

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        # Compute the L_inf norm along the second dimension (L)
        true_norm = torch.max(torch.abs(true_reshaped), dim=1)[0]  # (batch_size, out_dim)
        pred_diff_norm = torch.max(torch.abs(true_reshaped - pred_reshaped), dim=1)[0]  # (batch_size, out_dim)

        # Compute the relative error
        rel_error = pred_diff_norm / true_norm  # (batch_size, out_dim)

        # Average across batch and out_dim
        return torch.sum(torch.mean(rel_error, dim=-1)) # average over variables, sum over the batch

#loss function with relative Lp loss
class RelLpNorm(object):
    def __init__(self, out_dim, p):
        super(RelLpNorm, self).__init__()
        self._out_dim = out_dim
        self._ord     = p

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.reshape(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.reshape(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        # Compute the L2 norm along the second dimension (L)
        true_norm = torch.norm(true_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)
        pred_diff_norm = torch.norm(true_reshaped - pred_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)

        # Compute the relative error
        rel_error = pred_diff_norm / true_norm  # (batch_size, out_dim)

        # Average across batch and out_dim
        return torch.sum(torch.mean(rel_error, dim=-1)) # average over variables, sum over the batch

