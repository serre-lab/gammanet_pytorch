import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()
plt.show()

def real_number_batch_to_indexes(real_numbers, bins):
    """Converts a batch of real numbers to a batch of one hot vectors for the bins the real numbers fall in."""
    _, indexes = (real_numbers.view(-1, 1) - bins.view(1, -1)).abs().min(dim=1)
    return indexes

def real_to_index(real_numbers, n_classes):
    return (real_numbers*n_classes).to(torch.int)

def onehot_to_real(onehots):
    n_classes = onehots.shape[-1] 
    indexes = onehots.max(-1)[0]
    real_numbers = indexes/n_classes + indexes.clone().uniform_(0,1/n_classes) #torch.FloatTensor(indexes.shape)
    return real_numbers

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("biasqq" not in n) and (p.grad is not None):
            
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if p.grad.abs().mean() == 0:
                layers.append(n + "ZERO")
            elif p.grad.abs().mean() < 0.00001:
                layers.append(n + "SMALL")
            else:
                layers.append(n)
    
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.draw()
    plt.pause(0.001)
    return 0

def conv2d_same_padding(input, weight, bias=None, stride=(1,1), padding=(1,1), dilation=(1,1), groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

class Conv2dSamePadding(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_nl(name, fun=False, **kwargs):
    if hasattr(F, name) and fun:
        return getattr(F, name)
    elif hasattr(nn, name):
        return getattr(nn, name)(**kwargs)
    else:
        raise Exception("non-linearity doesn't exist")

def get_norm(name, **kwargs):
    if hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise Exception("normalization doesn't exist")