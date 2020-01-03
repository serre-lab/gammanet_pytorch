import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init


from layers.hgru_base import hConvGRUCell


class VGG_16(nn.Module):
    def __init__(self, weights_path, load_weights=True, freeze_layers=True):
        super().__init__()
        self.layers = [ 'conv1_1','conv1_2','maxpool1',
                        'conv2_1','conv2_2','maxpool2',
                        'conv3_1','conv3_2','conv3_3','maxpool3',
                        'conv4_1','conv4_2','conv4_3','maxpool4',
                        'conv5_1','conv5_2','conv5_3']

        self.filters = [64,      64,      64,      
                        128,     128,     128,     
                        256,     256,     256,    256,
                        512,     512,     512,    512,
                        512,     512,     512]
        
        self.build_layers()
        
        self.weights_path = weights_path

        if load_weights:
            self.load_state_dict(torch.load(weights_path),strict=False)
            #self.load_layers()
        if freeze_layers:
            self.freeze_layers()  
        
    
    def build_layers(self):
        #vgg model
        self.conv1_1 = nn.Conv2d(  3,  64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # self.model_layers = nn.ModuleDict({
        #     'conv1_1': self.conv1_1,
        #     'conv1_2': self.conv1_2,
            
        #     'maxpool1': self.maxpool1,
            
        #     'conv2_1': self.conv2_1,
        #     'conv2_2': self.conv2_2,

        #     'maxpool2': self.maxpool2,
            
        #     'conv3_1': self.conv3_1,
        #     'conv3_2': self.conv3_2,
        #     'conv3_3': self.conv3_3,

        #     'maxpool3': self.maxpool3,

        #     'conv4_1': self.conv4_1,
        #     'conv4_2': self.conv4_2,
        #     'conv4_3': self.conv4_3,
            
        #     'maxpool4': self.maxpool4,

        #     'conv5_1': self.conv5_1,
        #     'conv5_2': self.conv5_2,
        #     'conv5_3': self.conv5_3          
        # })

        # self.vgg_layers = nn.ModuleDict({
        #     'conv1_1': nn.Conv2d(  3,  64, kernel_size=3, padding=1),
        #     'conv1_2': nn.Conv2d( 64,  64, kernel_size=3, padding=1),
        
        #     'maxpool1': nn.MaxPool2d(kernel_size=2, stride=2),

        #     'conv2_1': nn.Conv2d( 64, 128, kernel_size=3, padding=1),
        #     'conv2_2': nn.Conv2d( 64, 128, kernel_size=3, padding=1),
        
        #     'maxpool2': nn.MaxPool2d(kernel_size=2, stride=2),

        #     'conv3_1': nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     'conv3_2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     'conv3_3': nn.Conv2d(256, 256, kernel_size=3, padding=1),
        
        #     'maxpool3': nn.MaxPool2d(kernel_size=2, stride=2),

        #     'conv4_1': nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     'conv4_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     'conv4_3': nn.Conv2d(512, 512, kernel_size=3, padding=1),
        
        #     'maxpool4': nn.MaxPool2d(kernel_size=2, stride=2),

        #     'conv5_1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     'conv5_2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     'conv5_3': nn.Conv2d(512, 512, kernel_size=3, padding=1)            
        # })

    def freeze_layers(self,layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = False
                getattr(self,l).bias.requires_grad = False

    def unfreeze_layers(self, layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = True
                getattr(self,l).bias.requires_grad = True

    def load_layers(self):
        
        weights = np.load(self.weights_path).item()
        for l in self.layers:
            if 'conv' in l:
                getattr(self,l).weight.data = torch.FloatTensor(weights[l]['weight'])
                getattr(self,l).bias.data = torch.FloatTensor(weights[l]['bias'])

    def forward(self, inputs):
        x = inputs

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.maxpool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        
        x = self.maxpool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))

        x = self.maxpool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = self.maxpool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        return x



