
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from utils.pt_utils import Conv2dPad

# input: [0,1] in (CHW) RGB

# transforms.Resize(256),
# transforms.CenterCrop(224),
# transforms.ToTensor(),
# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = Conv2dPad(inplanes, squeeze_planes, kernel_size=1, padding_mode='reflect')
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = Conv2dPad(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, padding_mode='reflect')
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = Conv2dPad(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, padding_mode='reflect')
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def load(self, weights):
        self.squeeze.weight.data    = torch.FloatTensor(weights['squeeze']['weight'])
        self.squeeze.bias.data      = torch.FloatTensor(weights['squeeze']['bias'])

        self.expand1x1.weight.data  = torch.FloatTensor(weights['expand1x1']['weight'])
        self.expand1x1.bias.data    = torch.FloatTensor(weights['expand1x1']['bias'])
        
        self.expand3x3.weight.data  = torch.FloatTensor(weights['expand3x3']['weight'])
        self.expand3x3.bias.data    = torch.FloatTensor(weights['expand3x3']['bias'])

    def freeze(self):
        self.squeeze.weight.requires_grad =False
        self.squeeze.bias.requires_grad =False
        
        self.expand1x1.weight.requires_grad =False
        self.expand1x1.bias.requires_grad =False
        
        self.expand3x3.weight.requires_grad =False
        self.expand3x3.bias.requires_grad =False

    def unfreeze(self):
        self.squeeze.weight.requires_grad =True
        self.squeeze.bias.requires_grad =True
        
        self.expand1x1.weight.requires_grad =True
        self.expand1x1.bias.requires_grad =True
        
        self.expand3x3.weight.requires_grad =True
        self.expand3x3.bias.requires_grad =True

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([
            F.relu(self.expand1x1(x)),
            F.relu(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, weight_path=None, load_weights=True, freeze_layers=False, n_layers=12):
        super().__init__()
        self.layers = ['conv1', 'pool1', 
                        'fire1', 'fire2', 'pool2', 
                        'fire3', 'fire4', 'pool3', 
                        'fire5', 'fire6', 'fire7', 'fire8']
        self.layers = self.layers[:n_layers]
        
        self.filters = [64, 64,
                        128, 128, 128,
                        256, 256, 256,
                        384, 384, 512, 512]

        self.filters = self.filters[:n_layers]
        
        self.weight_path = weight_path
        self.build_layers()

        if load_weights:
            self.load_state_dict(torch.load(weight_path), strict=False)
            #self.load_layers()
        if freeze_layers:
            self.freeze_layers()
        

    def build_layers(self):

        prev_fan = 3 
        for l, f in zip(self.layers, self.filters):
            if 'conv' in l:
                setattr(self, l, Conv2dPad(prev_fan, f, kernel_size=3, stride=2, padding_mode='reflect'))
            elif 'pool' in l:
                setattr(self, l, nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            elif 'fire' in l:
                setattr(self, l, Fire(prev_fan, f//8, f//2, f//2))
            prev_fan = f

        # self.conv1 =    nn.Conv2d(3, 64, kernel_size=3, stride=2)
        # self.pool1 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.fire1 =    Fire(64, 16, 64, 64)
        # self.fire2 =    Fire(128, 16, 64, 64)
        # self.pool2 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.fire3 =    Fire(128, 32, 128, 128)
        # self.fire4 =    Fire(256, 32, 128, 128)
        # self.pool3 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # self.fire5 =    Fire(256, 48, 192, 192)
        # self.fire6 =    Fire(384, 48, 192, 192)
        # self.fire7 =    Fire(384, 64, 256, 256)
        # self.fire8 =    Fire(512, 64, 256, 256)    
    
    def load_layers(self, weight_path=None):
        if weight_path is None:
            weight_path = self.weight_path
        assert weight_path is not None, 'file path unspecified'
        
        weights = np.load(weight_path).item()
        
        for l in self.layers:
            if 'conv' in l:
                getattr(self,l).weight.data = torch.FloatTensor(weights[l]['weight'])
                getattr(self,l).bias.data = torch.FloatTensor(weights[l]['bias'])
            elif 'fire' in l:
                getattr(self,l).load(weights[l])
        
        # self.conv1.weight.data = torch.FloatTensor(weights['conv1']['weight'])
        # self.conv1.bias.data = torch.FloatTensor(weights['conv1']['bias'])

        # self.fire1.load(weights['fire1'])
        # self.fire2.load(weights['fire2'])
        # self.fire3.load(weights['fire3'])
        # self.fire4.load(weights['fire4'])
        # self.fire5.load(weights['fire5'])
        # self.fire6.load(weights['fire6'])
        # self.fire7.load(weights['fire7'])
        # self.fire8.load(weights['fire8'])

    def freeze_layers(self,layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = False
                getattr(self,l).bias.requires_grad = False
            elif 'fire' in l:
                getattr(self,l).freeze()

    def unfreeze_layers(self, layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = True
                getattr(self,l).bias.requires_grad = True
            elif 'fire' in l:
                getattr(self,l).unfreeze()

    def forward(self, x):
        # x = (x - self.mean_const)/ self.std_const

        for l in self.layers:
            if 'conv' in l:
                x = F.relu(getattr(self,l)(x))
            else:
                x = getattr(self,l)(x)

        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        # x = self.fire1(x)
        # x = self.fire2(x)
        # x = self.pool2(x)
        # x = self.fire3(x)
        # x = self.fire4(x)
        # x = self.pool3(x)
        # x = self.fire5(x)
        # x = self.fire6(x)
        # x = self.fire7(x)
        # x = self.fire8(x)
        
        return x