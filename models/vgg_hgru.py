import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from layers.hgru_base import hConvGRUCell


class VGGhGRU(nn.Module):
    def __init__(self, vgg_weights_path, load_weights=True, filter_size=9, timesteps=6):
        super().__init__()
        self.vgg_weights_path = vgg_weights_path
        self.timesteps = timesteps
        
        if isinstance(filter_size,int):
            self.filter_size = [filter_size]*4
        print('building vgg layers')
        self.build_vgg_layers()
        print('building hgru units')
        self.build_hgru_units()
        print('building readout layers')
        self.build_readout()

        if load_weights:
            self.load_vgg_weights()

        
    def build_hgru_units(self):
        
        maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1 = maxpool1
        self.unit1 = hConvGRUCell(128, 128, self.filter_size[0])
        self.unit1.train()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.unit2 = hConvGRUCell(256, 256, self.filter_size[1])
        self.unit2.train()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.unit3 = hConvGRUCell(512, 512, self.filter_size[2])
        self.unit3.train()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.unit4 = hConvGRUCell(512, 512, self.filter_size[3])
        self.unit4.train()
    
    def build_vgg_layers(self):
        #vgg model
        # self.conv1_1 = nn.Conv2d(  3,  64, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1)
        # # nn.MaxPool2d(kernel_size=2, stride=2) # five pooling layers
        # self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.vgg_layers = nn.ModuleDict({
            'conv1_1': nn.Conv2d(  3,  64, kernel_size=3, padding=1),
            'conv1_2': nn.Conv2d( 64,  64, kernel_size=3, padding=1),

            'conv2_1': nn.Conv2d( 64, 128, kernel_size=3, padding=1),
            'conv2_2': nn.Conv2d( 64, 128, kernel_size=3, padding=1),

            'conv3_1': nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'conv3_2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'conv3_3': nn.Conv2d(256, 256, kernel_size=3, padding=1),

            'conv4_1': nn.Conv2d(256, 512, kernel_size=3, padding=1),
            'conv4_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'conv4_3': nn.Conv2d(512, 512, kernel_size=3, padding=1),

            'conv5_1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'conv5_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'conv5_3': nn.Conv2d(512, 512, kernel_size=3, padding=1)            
        })
        for k,v in self.vgg_layers.items():
            v.weight.requires_grad =False
            v.bias.requires_grad =False
    
    def load_vgg_weights(self):
        
        vgg_weights = np.load(self.vgg_weights_path).item()
        for k in self.vgg_layers:
            self.vgg_layers[k].weight.data = torch.FloatTensor(vgg_weights[k]['weight'])
            self.vgg_layers[k].bias.data = torch.FloatTensor(vgg_weights[k]['bias'])

    def build_readout(self):
        self.last_conv = nn.Conv2d(512, 128, kernel_size=1, padding=1)
        # global average mean layer is replaced by mean op in forward
        # self.bn = nn.BatchNorm2d(25, eps=1e-03)
        
        self.fc = nn.Linear(128, 3)
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, inputs, return_hidden=False,return_error=False):
        
        hidden_1 = None
        hidden_2 = None
        hidden_3 = None
        hidden_4 = None
        l_h = []
        l_e = []

        x = inputs

        x = F.relu(self.vgg_layers['conv1_1'](x))
        x = F.relu(self.vgg_layers['conv1_2'](x))
        x = self.maxpool1(x)
        x = F.relu(self.vgg_layers['conv2_1'](x))
        x = F.relu(self.vgg_layers['conv2_2'](x))

        conv_input = x
        for ts in range(self.timesteps):
            x = conv_input
            hidden_1, error_1 = self.unit1(x, hidden_1,ts)
            x = self.maxpool2(hidden_1)
            
            x = F.relu(self.vgg_layers['conv3_1'](x))
            x = F.relu(self.vgg_layers['conv3_2'](x))
            x = F.relu(self.vgg_layers['conv3_3'](x))

            hidden_2, error_2 = self.unit2(x, hidden_2,ts)
            x = self.maxpool3(hidden_2)
            
            x = F.relu(self.vgg_layers['conv4_1'](x))
            x = F.relu(self.vgg_layers['conv4_2'](x))
            x = F.relu(self.vgg_layers['conv4_3'](x))

            hidden_3, error_3 = self.unit3(x, hidden_3,ts)
            x = self.maxpool4(hidden_3)
            
            x = F.relu(self.vgg_layers['conv5_1'](x))
            x = F.relu(self.vgg_layers['conv5_2'](x))
            x = F.relu(self.vgg_layers['conv5_3'](x))

            hidden_4, error_4 = self.unit3(x, hidden_4,ts)

            if return_hidden:
                l_h.append(torch.stack([hidden_1,hidden_2,hidden_3,hidden_4],dim=1))
            if return_error:
                l_e.append(torch.stack([error_1,error_2,error_3,error_4],dim=1))

            
        x = self.last_conv(hidden_4)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        output = torch.sigmoid(self.fc(x))
        if return_hidden:
            torch.stack(l_h,dim=1)
        if return_error:
            torch.stack(l_e,dim=1)
        return output, (l_h,l_e)

