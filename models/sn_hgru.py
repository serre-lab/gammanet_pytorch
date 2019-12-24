
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from models.squeezenet import Fire, SqueezeNet
from layers.hgru_base import hConvGRUCell


class SN_hGRU(nn.Module):

    def __init__(self, weight_path=None, load_weights=True, freeze_layers=False, add_hgru=[],timesteps=6):
        super().__init__()
        self.layers = ['conv1', 'pool1', 'fire1', 'fire2', 'pool2', 'fire3', 'fire4', 'pool3', 'fire5', 'fire6', 'fire7', 'fire8']
        self.filters = [64,      64,      64,      128,     128,     128,     256,     256,     256,     384,     384,     512]
        self.add_hgru = add_hgru
        self.timesteps = timesteps
        self.weight_path = weight_path
        self.build_sn_layers()
        #self.sn = SqueezeNet()
        if add_hgru is not None and add_hgru!= []:
            assert timesteps is not None, 'number of timesteps unspecified'
            self.build_hgru_units()

        self.mean = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None]
        self.std = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None]
        # self.mean.requires_grad=False
        # self.std.requires_grad=False
        self.register_buffer('mean_const', self.mean)
        self.register_buffer('std_const', self.std)

        if load_weights:
            self.load_state_dict(torch.load(weight_path),strict=False)
            #self.load_layers()
        if freeze_layers:
            self.freeze_layers()        

    def build_sn_layers(self):
        self.conv1 =    nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.pool1 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire1 =    Fire(64, 16, 64, 64)
        self.fire2 =    Fire(128, 16, 64, 64)
        # hgru_1
        self.pool2 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire3 =    Fire(128, 32, 128, 128)
        self.fire4 =    Fire(256, 32, 128, 128)
        # hgru_2
        self.pool3 =    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire5 =    Fire(256, 48, 192, 192)
        self.fire6 =    Fire(384, 48, 192, 192)
        self.fire7 =    Fire(384, 64, 256, 256)
        self.fire8 =    Fire(512, 64, 256, 256)
        # hgru_3
        self.sn_layers = [self.conv1,self.pool1,
                          self.fire1,self.fire2,self.pool2,
                          self.fire3,self.fire4,self.pool3,
                          self.fire5,self.fire6,self.fire7,self.fire8]
        self.sn_layers = nn.ModuleList(self.sn_layers)

    def build_hgru_units(self):
        self.hgru_units = []
        self.hgru_positions = []
        for pos,f_size in self.add_hgru:
            self.hgru_positions.append(self.layers.index(pos))
            f = self.filters[self.layers.index(pos)]
            unit = hConvGRUCell(f, f, f_size)
            unit.train()
            self.hgru_units.append(unit)
        init_pos = self.layers.index(self.add_hgru[0][0])
        self.hgru_units = nn.ModuleList(self.hgru_units)
        self.input_layers = self.sn_layers[1:init_pos+1]
        self.recur_layers = self.sn_layers[init_pos+1:]
    
    def load_layers(self, weight_path=None):
        if weight_path is None:
            weight_path = self.weight_path
        assert weight_path is not None, 'file path unspecified'
        
        weights = np.load(weight_path).item()
        
        self.conv1.weight.data = torch.FloatTensor(weights['conv1']['weight'])
        self.conv1.bias.data = torch.FloatTensor(weights['conv1']['bias'])

        self.fire1.load(weights['fire1'])
        self.fire2.load(weights['fire2'])
        self.fire3.load(weights['fire3'])
        self.fire4.load(weights['fire4'])
        self.fire5.load(weights['fire5'])
        self.fire6.load(weights['fire6'])
        self.fire7.load(weights['fire7'])
        self.fire8.load(weights['fire8'])

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

    def forward(self, x, return_hidden=False):
        x = (x - self.mean_const)/ self.std_const
        
        if len(self.hgru_units) == 0:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = self.fire1(x)
            x = self.fire2(x)
            x = self.pool2(x)
            x = self.fire3(x)
            x = self.fire4(x)
            x = self.pool3(x)
            x = self.fire5(x)
            x = self.fire6(x)
            x = self.fire7(x)
            x = self.fire8(x)
            return x
        else:
            
            hgru_hidden = [None]*len(self.hgru_units)
            last_hidden = []

            conv_input = F.relu(self.conv1(x))
            
            for l in self.input_layers:
                conv_input = l(conv_input)
            
            for i in range(self.timesteps):
                hidden, _ = self.hgru_units[0](conv_input,hgru_hidden[0],timestep=i)
                hgru_hidden[0] = hidden
                x = hidden
                hgru_i = 1
                for j,l in enumerate(self.recur_layers):
                    x = l(x)
                    if j+len(self.input_layers)+1 in self.hgru_positions:
                        hidden, _ = self.hgru_units[hgru_i](x, hgru_hidden[hgru_i], timestep=i)
                        hgru_hidden[hgru_i] = hidden
                        x = hidden
                        hgru_i+=1
                if return_hidden:
                    last_hidden.append(x)
            
            if return_hidden:
                last_hidden = torch.stack(last_hidden,dim=1)
            return hgru_hidden[-1], last_hidden