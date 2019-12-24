import torch
import numpy as np
import torch.nn.functional as F
from utils import pt_utils

from collections import OrderedDict
def get_eval(metric):

    #if isinstance(cfg.training.eval,list):
    if metric in globals():
        return globals()[metric]
    else:
        print('evaluation score not found')

def mean_CI(predicted_XYZ, ideal_XYZ, XYZ=None, **kwargs):
    
    if XYZ is None:
        return torch.Tensor([0])
    if predicted_XYZ.shape[-1] != 3:
        predicted_XYZ = pt_utils.onehot_to_real(predicted_XYZ)
    test_XYZ = XYZ
    assert (ideal_XYZ.shape[-1], test_XYZ.shape[-1], predicted_XYZ.shape[-1]) == (3,3,3), 'incorrect shape'

    predicted_XYZ = predicted_XYZ.cpu()

    ideal_xyz = ideal_XYZ / ideal_XYZ.sum(-1, keepdims=True)
    test_xyz = test_XYZ / test_XYZ.sum(-1, keepdims=True)
    predicted_xyz = predicted_XYZ / predicted_XYZ.sum(-1, keepdims=True)

    A = torch.norm(ideal_xyz[..., :2] - test_xyz[..., :2], p=2,dim=-1)
    B = torch.norm(ideal_xyz[..., :2] - predicted_xyz[..., :2], p=2,dim=-1)
    #C = torch.norm(test_xyz[..., :2] - predicted_xyz[..., :2], p=2,dim=-1)

    CI = 1 - B/A
    CI_mean = torch.mean(CI)
    
    return CI_mean
    
def constancy_score(predicted_XYZ, ideal_XYZ, XYZ, get_stats=False,**kwargs):
    if predicted_XYZ.shape[-1] != 3:
        predicted_XYZ = pt_utils.onehot_to_real(predicted_XYZ)
    test_XYZ = XYZ
    assert (ideal_XYZ.shape[-1], test_XYZ.shape[-1], predicted_XYZ.shape[-1]) == (3,3,3), 'incorrect shape'

    predicted_XYZ = predicted_XYZ.cpu()

    ideal_xyz = ideal_XYZ / ideal_XYZ.sum(-1, keepdims=True)
    test_xyz = test_XYZ / test_XYZ.sum(-1, keepdims=True)
    predicted_xyz = predicted_XYZ / predicted_XYZ.sum(-1, keepdims=True)

    A = torch.norm(ideal_xyz[..., :2] - test_xyz[..., :2], p=2,dim=-1)
    B = torch.norm(ideal_xyz[..., :2] - predicted_xyz[..., :2], p=2,dim=-1)
    C = torch.norm(test_xyz[..., :2] - predicted_xyz[..., :2], p=2,dim=-1)

    CI = 1 - B/A
    cosphi = (A**2 + C**2 - B**2) / (2* A * C)
    CI = 1 - B/A
    BR = C/A
    BRphi = C/A * cosphi

    result = OrderedDict([('CI', CI), ('BR', BR), ('BRphi', BRphi)])
    
    if get_stats:
        CI_mean, CI_sem = torch.mean(CI), CI.std(dim=-1)/torch.pow(CI.shape[-1],p=1/2)
        BR_mean, BR_sem = torch.mean(BR), BR.std(dim=-1)/torch.pow(BR.shape[-1],p=1/2)
        BRphi_mean, BRphi_sem = torch.mean(BRphi), BRphi.std(dim=-1)/torch.pow(BRphi.shape[-1],p=1/2)

        result.update(OrderedDict([
                        ('CI_mean', CI_mean),
                        ('CI_sem', CI_sem), 
                        ('BR_mean', BR_mean),
                        ('BR_sem', BR_sem), 
                        ('BRphi_mean', BRphi_mean),
                        ('BRphi_sem', BRphi_sem)]))

    return result


def neg_angular_error(estimate, illum_rgb, reduce=True, **kwargs):
    # Assume all vectors already have unit norm
    estimate = estimate.cpu()
    target = illum_rgb
    
    if len(estimate.shape)>2:
        estimate = estimate.view([-1,estimate.shape[-1]])
    if len(target.shape)>2:
        target = target.view([-1,target.shape[-1]])
        
    safe_v = 0.999999

    estimate = F.normalize(estimate, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    dot = torch.sum(estimate * target, 1)
    dot = torch.clamp(dot, -safe_v, safe_v)

    angular_error = torch.acos(dot) * (180 / np.pi)
    
    if reduce:
        angular_error = torch.mean(angular_error)
    return -angular_error

