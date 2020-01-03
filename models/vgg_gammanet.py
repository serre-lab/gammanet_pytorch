
#c_input-pool          readout
#  h------------------>td ------->
#  c-p                 us
#    h-------------->td --------->
#    c-p             us
#      h---------->td ----------->
#      c-p         us
#        h------>td ------------->
#        c-p     us
#            h ------------------>

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from models.vgg_16 import VGG_16

from layers.fgru_base import fGRUCell

from utils.pt_utils import Conv2dSamePadding
from utils import pt_utils

class VGG_16_GN(nn.Module):
    def __init__(self, 
                weights_path, 
                load_weights=True, 
                gn_params=[], 
                timesteps=6,
                filter_size=9, 
                hidden_init='identity',
                attention='gala', # 'se', None
                attention_layers=2,
                saliency_filter_size=5,
                normalization_fgru='InstanceNorm2d',
                normalization_fgru_params={},
                normalization_gate='InstanceNorm2d',
                normalization_gate_params={},
                force_alpha_divisive=True,
                force_non_negativity=True,
                multiplicative_excitation=True,
                ff_non_linearity='ReLU',
                us_resize_before_block=True):
        super().__init__()
        self.timesteps = timesteps
        self.gn_params = gn_params

        assert len(gn_params)%2==1, 'the number of fgrus is not odd'
        self.gn_layers = len(gn_params)//2 +1
        
        self.ff_non_linearity = ff_non_linearity
        self.normalization_fgru = normalization_fgru

        self.us_resize_before_block = us_resize_before_block

        self.normalization_fgru_params = normalization_fgru_params
        self.fgru_params = {
            'hidden_init'               : hidden_init,
            'attention'                 : attention,
            'attention_layers'          : attention_layers,
            'saliency_filter_size'      : saliency_filter_size,
            'normalization_fgru'        : normalization_fgru,
            'normalization_fgru_params' : normalization_fgru_params,
            'normalization_gate'        : normalization_gate,
            'normalization_gate_params' : normalization_gate_params,
            'ff_non_linearity'          : ff_non_linearity,
            'force_alpha_divisive'      : force_alpha_divisive,
            'force_non_negativity'      : force_non_negativity,
            'multiplicative_excitation' : multiplicative_excitation
        }
        self.base_ff = VGG_16(weights_path=weights_path, load_weights=load_weights)

        self.build_fb_layers()
    
    def build_fb_layers(self):

        self.h_units = []
        self.ds_blocks = []

        self.td_units = []
        self.us_blocks = []
        
        prev_pos = 0
        base_layers = self.base_ff.layers
        base_filters = self.base_ff.filters
        
        # downsampling and horizontal units
        for i in range(self.gn_layers):
            pos,k_size = self.gn_params[i]
            layer_pos = base_layers.index(pos)+1
            
            feats = base_filters[base_layers.index(pos)]

            h_unit = fGRUCell(input_size = feats, 
                            hidden_size = feats, 
                            kernel_size = k_size,
                            **self.fgru_params)
            h_unit.train()
            self.h_units.append(h_unit)

            self.ds_blocks.append(self.create_ds_block(base_layers[prev_pos:layer_pos]))
            prev_pos = layer_pos

        # last downsampling output block
        if layer_pos+1 < len(base_layers):

            # block_layers = [getattr(self.base_ff,k) for k in base_layers[prev_pos:len(base_layers)]]
            # nn.Sequential(*block_layers) if len(block_layers)!=1 else block_layers[0]
            self.output_block = self.create_ds_block(base_layers[prev_pos:len(base_layers)])
            
        td_feats = feats

        # upsampling and topdown units
        for i in range(self.gn_layers,len(self.gn_params)):
            pos,k_size = self.gn_params[i]
            layer_pos = base_layers.index(pos)+1
            
            feats = base_filters[base_layers.index(pos)]

            td_unit = fGRUCell(input_size = feats, 
                            hidden_size = feats, 
                            kernel_size = k_size,
                            **self.fgru_params)
            td_unit.train()
            self.td_units.append(td_unit)

            us_block = self.create_us_block(td_feats,feats)
            self.us_blocks.append(us_block)

            td_feats = feats

        self.input_block = self.ds_blocks.pop(0)

        self.h_units = nn.ModuleList(self.h_units)
        self.ds_blocks = nn.ModuleList(self.ds_blocks)

        self.td_units = nn.ModuleList(self.td_units)
        self.us_blocks = nn.ModuleList(self.us_blocks)

    def create_ds_block(self,base_layers):
        # ds block: depends on base_ff non_linearity and bn
        module_list = []
        for l in base_layers:
            module_list.append(getattr(self.base_ff,l))
            if 'conv' in l:
                module_list.append(nn.ReLU())
        
        return nn.Sequential(*module_list) if len(module_list)>1 else module_list[0]

    def create_us_block(self,input_feat, output_feat):
        # us options: norm top_h, resize before or after block, ...
        normalization_fgru = pt_utils.get_norm(self.normalization_fgru)
        
        module_list = [
            normalization_fgru(input_feat,**self.normalization_fgru_params),
            Conv2dSamePadding(input_feat,output_feat,1),
            nn.ReLU(),
            Conv2dSamePadding(output_feat,output_feat,1),
            nn.ReLU(),
        ]
        
        # bilinear resize -> dependent on the other size
        # other version : norm -> conv 1*1 -> norm -> (extra conv 1*1 ->) resize
        # other version : transpose_conv 4*4/2 -> conv 3*3 -> norm
        return nn.Sequential(*module_list)
        
    def us_block(self, block, input_, out_size):
        if self.us_resize_before_block:
            input_ = F.interpolate(input_,out_size, mode='bilinear')
            output = block(input_) 
        else:
            input_ = block(input_)
            output = F.interpolate(input_,out_size, mode='bilinear') 
        
        return output

    def forward(self, inputs, return_hidden=False):
        x = inputs
        h_hidden = [None] * len(self.h_units)
        
        conv_input = self.input_block(x)

        last_hidden = []

        for i in range(self.timesteps):
            x = conv_input

            # down_sample
            for l in range(len(self.h_units)):
                hidden, _ = self.h_units[l](x, h_hidden[l], timestep=i)
                h_hidden[l] = hidden
                if l<len(self.ds_blocks):
                    x = self.ds_blocks[l](hidden)
                else:
                    x = hidden
            
            # up_sample
            for l,h in enumerate(reversed(h_hidden[:-1])):
                x = self.us_block(self.us_blocks[l], x, h.shape[2:])
                x, _ = self.td_units[l](h, x, timestep=i)

                x += h
                h_hidden[len(self.td_units)-l-1] = x


            if return_hidden:
                last_hidden.append(x)

        if return_hidden:
            last_hidden = torch.stack(last_hidden,dim=1)
            if hasattr(self, 'output_block'):
                in_shape = last_hidden.shape.tolist()
                last_hidden = self.output_block(last_hidden.view([-1]+in_shape[2:]))
                out_shape = last_hidden.shape.tolist()
                last_hidden = last_hidden.view(in_shape[:2] + out_shape[1:])

        if hasattr(self, 'output_block'):
            x = self.output_block(x)

        return x, last_hidden