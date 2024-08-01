import operator
from functools import reduce
import torch
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()))
    return c

class MaxNorm(object):
    def __init__(self, out_dim):
        super(RelMaxNorm, self).__init__()
        self._out_dim = out_dim

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        # Compute the L_inf norm along the second dimension (L)
        pred_diff_norm = torch.max(torch.abs(true_reshaped - pred_reshaped), dim=1) / true_reshaped.shape[1]  # (batch_size, out_dim)

        # Average across batch and out_dim
        return torch.sum(torch.mean(pred_diff_norm, dim=-1)) # average over variables, sum over the batch

#loss function with relative Lp loss
class LpNorm(object):
    def __init__(self, out_dim, p):
        super(LpNorm, self).__init__()
        self._out_dim = out_dim
        self._ord     = p
        if self._ord  == 1:
            self.loss_func = torch.nn.L1Loss(reduction='none')
        elif self._ord  == 2:
            self.loss_func = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        # Compute the L2 norm along the second dimension (L)
        # pred_diff_norm = torch.norm(true_reshaped - pred_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)
        pred_diff_norm = self.loss_func(true_reshaped, pred_reshaped)
        # Average across batch and out_dim
        return torch.sum(torch.mean(pred_diff_norm, dim=(1,2))) # average over variables, sum over the batch

class L01Norm(object):
    def __init__(self, out_dim):
        super(RelL01Norm, self).__init__()
        self._out_dim = out_dim

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        abs_err       = torch.abs(true_reshaped - pred_reshaped)
        weights       = torch.nn.Softmax(dim=1)(abs_err)
        weighted_err  = abs_err * weights
        l01_loss      = torch.sum(weighted_err, dim=1)
        
        return torch.sum(torch.mean(l01_loss, dim=-1))

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
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        # Compute the L2 norm along the second dimension (L)
        true_norm = torch.norm(true_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)
        pred_diff_norm = torch.norm(true_reshaped - pred_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)

        # Compute the relative error
        rel_error = pred_diff_norm / true_norm  # (batch_size, out_dim)

        # Average across batch and out_dim
        return torch.sum(torch.mean(rel_error, dim=-1)) # average over variables, sum over the batch

class RelL01Norm(object):
    def __init__(self, out_dim):
        super(RelL01Norm, self).__init__()
        self._out_dim = out_dim

    def __call__(self, true, pred):
        # Reshape true and pred
        true_reshaped = true.view(true.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)
        pred_reshaped = pred.view(pred.size(0), -1, self._out_dim)  # (batch_size, L, out_dim)

        abs_err       = torch.abs(true_reshaped - pred_reshaped)
        weights       = torch.nn.Softmax(dim=1)(abs_err)
        weighted_err  = abs_err * weights
        l01_loss      = torch.sum(weighted_err, dim=1)
        
        abs_true      = torch.abs(true_reshaped)
        weights_true  = torch.nn.Softmax(dim=1)(abs_true)
        weighted_true = abs_true * weights_true
        l01_true      = torch.sum(weighted_true, dim=1)

        rel_l01_loss  = l01_loss / l01_true
        return torch.sum(torch.mean(rel_l01_loss, dim=-1))
        # Compute the L2 norm along the second dimension (L)
        # true_norm = torch.norm(true_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)
        # pred_diff_norm = torch.norm(true_reshaped - pred_reshaped, p=self._ord, dim=1)  # (batch_size, out_dim)

        # # Compute the relative error
        # rel_error = pred_diff_norm / true_norm  # (batch_size, out_dim)

        # Average across batch and out_dim
        # return torch.sum(torch.mean(rel_error, dim=-1)) # average over variables, sum over the batch
