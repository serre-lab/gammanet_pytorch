
import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
from utils import pt_utils

def get_loss(loss_name, **kwargs):
    if loss_name in globals():
        return globals()[loss_name]
    elif hasattr(F, loss_name):
        return getattr(F, loss_name)
    elif hasattr(nn, loss_name):
        return getattr(nn, loss_name)(**kwargs)
    else:
        print("loss function doesn't exist")

def class_balanced_bce_with_logits(input_, target, gamma=0.5,reduction='mean'):
    target = torch.where(target>=0.5,torch.ones_like(target),target)
    ones = (target==1)
    zeros = (target==0)
    n_ones = ones.sum()
    n_zeros = zeros.sum()
    combined = n_ones+n_zeros
    ones = ones * (n_zeros * 1./ combined)
    zeros = zeros * (n_ones * 1.1/ combined) 
    weights = ones + zeros
    out = F.binary_cross_entropy_with_logits(input_, target, reduction='none')
    out = out * weights
    if reduction=='mean':
        out = out.mean()
    return out

def class_balanced_bce_with_logits_old(input_, target, gamma=0.5,reduction='mean'):
    # target = torch.where(target>0.5,torch.ones_like(target),target)
    
    neg_labels = torch.where(target<=0.0,torch.ones_like(target),torch.zeros_like(target))
    pos_labels = torch.where(target>gamma,torch.ones_like(target),torch.zeros_like(target))

    mask = neg_labels + pos_labels

    # ones = (target==1)
    # zeros = (target==0)
    n_ones = pos_labels.sum()
    n_zeros = (1-pos_labels).sum()

    beta = n_zeros/(n_zeros+n_ones)

    pos_weight = beta / (1 - beta)

    # combined = n_ones+n_zeros
    # ones = ones * (n_zeros * 1./ combined)
    # zeros = zeros * (n_ones * 1./ combined) 
    # weights = ones + zeros
    out = F.binary_cross_entropy_with_logits(input_, pos_labels, reduction='none')
    out = out * pos_weight
    out *= mask
    out *= (1-beta)

    if reduction=='mean':
        out = out.mean()
    return out
    
    

def pixel_error(input_,target,reduction='mean'):
    input_ = torch.sigmoid(input_)
    input_ = torch.where(input_>0.5,torch.ones_like(input_),torch.zeros_like(input_))
    error = torch.where((input_ != target),torch.ones_like(input_),torch.zeros_like(input_))
    error = error.view([error.shape[0],-1]).sum(-1)
    if reduction=='mean':
        error = error.mean()
    return error

def xyz_cross_entropy(output, target, **kwargs):
    """
        output: class predictions for X Y Z values. shape: [B,3,C] or [B,3,H,W,C]
        target: groundtruth category for X Y Z calues. shape: [B,3] or [B,3,H,W] 
    """
    target_shape = target.shape
    output_shape = output.shape
    if len(target_shape)>2:
        target = target.view(-1,3)
        output = output.view(-1,3,output_shape[-1])
        
    target = pt_utils.real_to_index(target, output_shape[-1])
    
    X_loss = F.cross_entropy(output[:,0,:], target[:,0].long(), **kwargs)
    Y_loss = F.cross_entropy(output[:,1,:], target[:,1].long(), **kwargs)
    Z_loss = F.cross_entropy(output[:,2,:], target[:,2].long(), **kwargs)
    
    return (X_loss+Y_loss+Z_loss)/3

def correlation_loss(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xm = F.normalize(xm, p=2, dim=1)
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xmt = F.normalize(xmt, p=2, dim=1)
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    #loss = loss/(4*d*d)

    return loss

def cosine_loss(estimate, target, reduce=True):
    # Assume all vectors already have unit norm
    estimate = F.normalize(estimate, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosines = torch.sum(estimate * target, 1)
    loss = 1 - cosines
    #angular_error = 180 * torch.acos(torch.clamp(cosines, -1, 1)) / 3.141592653589793
    if reduce:
        loss = torch.mean(loss)
        #angular_error = torch.mean(angular_error)
    return loss


def fc4_loss(per_patch_estimate, illums,per_patch_weight):
    global_loss = get_angular_loss(per_patch_estimate.sum((1, 2)), illums)
    loss = global_loss

def angular_loss(estimate, target, reduce=True):
    
    if len(estimate.shape)>2:
        estimate = estimate.view([-1,estimate.shape[-1]])
    if len(target.shape)>2:
        target = target.view([-1,target.shape[-1]])
    safe_v = 0.999999

    estimate = F.normalize(estimate, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    dot = torch.sum(estimate * target, 1)
    dot = torch.clamp(dot, -safe_v, safe_v)

    angle = torch.acos(dot) * (180 / np.pi)
    if reduce:
        angle = torch.mean(angle)

    return angle

if __name__ == "__main__":

    #print('XYZCrossEntroy' in dir())
    a = torch.randn([2,10,10])
    b = torch.where(a>0,torch.ones_like(a), torch.zeros_like(a))
    print(class_balanced_bce_with_logits(a,b))
    # print(globals()['XYZCrossEntroy'])
    # print('cross_entropy' in dir())
    #print()