import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

from utils.pt_utils import conv2d_same_padding, Conv2dSamePadding
from utils import pt_utils

# TODO symmetric init for conv_c1_w and conv_c2_w | maybe already done
# TODO bias init for gates (chronos)
# TODO check bn init 
# TODO add homunculus
# TODO add different normalization types
# TODO try varying hidden size (make it independent from input)

# TODO code attention and add it in g1 | done
# TODO init types for hidden (identity, zero, xavier) | done
# TODO solve padding issue | done


class fGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, 
                input_size, 
                hidden_size, 
                kernel_size,
                hidden_init='identity',
                attention='gala', # 'se', None
                attention_layers=2,
                # attention_normalization=True,
                saliency_filter_size=5,
                normalization_fgru='InstanceNorm2d',
                normalization_fgru_params={},
                normalization_gate='InstanceNorm2d',
                normalization_gate_params={},
                ff_non_linearity='ReLU',
                force_alpha_divisive=True,
                force_non_negativity=True,
                multiplicative_excitation=True):
        super().__init__()
        
        self.padding = 'same' # kernel_size // 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_init = hidden_init

        self.normalization_fgru = normalization_fgru
        self.normalization_gate = normalization_fgru
        self.normalization_fgru_params = normalization_fgru_params if normalization_fgru_params is not None else {}
        self.normalization_gate_params = normalization_gate_params if normalization_gate_params is not None else {}

        self.normalization_gate = normalization_gate

        normalization_fgru = pt_utils.get_norm(normalization_fgru)
        normalization_gate = pt_utils.get_norm(normalization_gate)

        self.force_alpha_divisive = force_alpha_divisive
        self.force_non_negativity = force_non_negativity
        self.multiplicative_excitation = multiplicative_excitation

        # add attention
        if attention is not None and attention_layers>0:
            if attention == 'se':
                self.attention = SE_Attention(  hidden_size, hidden_size, 1,
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            elif attention == 'gala':
                self.attention = GALA_Attention(hidden_size, hidden_size, saliency_filter_size, 
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            else:
                raise 'attention type unknown'
        else:
            self.conv_g1_w = nn.Parameter(torch.empty(hidden_size , hidden_size, 1, 1))
            init.orthogonal_(self.conv_g1_w)

        self.conv_g1_b = nn.Parameter(torch.empty(hidden_size,1,1))

        if self.normalization_gate:
            self.bn_g1 = normalization_gate(hidden_size, **self.normalization_gate_params)
            # init.constant_(self.bn_g1.weight, 0.1)

        if self.normalization_fgru:
            self.bn_c1 = normalization_fgru(hidden_size, **self.normalization_fgru_params)
            #init.constant_(self.bn_c1.weight, 0.1)

        self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))

        self.conv_g2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , 1, 1))
        self.conv_g2_b = nn.Parameter(torch.empty(hidden_size,1,1))
        
        if self.normalization_gate:
            self.bn_g2 = normalization_gate(hidden_size, **self.normalization_gate_params)
            #init.constant_(self.bn_g2.weight, 0.1)

        if self.normalization_fgru:
            self.bn_c2 = normalization_fgru(hidden_size, **self.normalization_fgru_params)
            #init.constant_(self.bn_c2.weight, 0.1)
        
        self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        init.orthogonal_(self.conv_c1_w)
        init.orthogonal_(self.conv_c2_w)

        self.conv_c1_w.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.conv_c2_w.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)

        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu = nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.multiplicative_excitation:
            self.omega = nn.Parameter(torch.empty((hidden_size,1,1)))
            self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.mu, 1)
        if self.multiplicative_excitation:
            init.constant_(self.omega, 1.0)
            init.constant_(self.kappa, 0.5)
        

    def forward(self, input_, prev_state2, timestep=0):
        if timestep == 0 and prev_state2 is None:
            prev_state2 = torch.empty_like(input_)
            if self.hidden_init =='identity':
                prev_state2 = input_
            elif self.hidden_init =='zero':
                init.zeros_(prev_state2)
            else:
                init.xavier_normal_(prev_state2)

        i = timestep
        
        h2 = prev_state2
        ############## circuit input
        
        if self.attention is not None:
            g1 = self.attention(h2)
        else:
            g1 = conv2d_same_padding(h2, self.conv_g1_w)
        
        # g1_intermediate
        if self.normalization_gate:
            g1 = self.bn_g1(g1)
        
        h2 = h2 * F.sigmoid(g1 + self.conv_g1_b)

        if self.normalization_fgru:
            self.bn_c1(h2)

        # c1 -> conv2d symmetric_weights, dilations
        c1 = conv2d_same_padding(h2,self.conv_c1_w)

        ############## input integration
        
        # alpha, mu
        if self.force_alpha_divisive:
            alpha = F.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        inh = (alpha * h2 + self.mu) * c1

        if self.force_non_negativity:
            h1 = F.relu(F.relu(input_) - F.relu(inh))
        else:
            h1 = F.relu(input_ - inh)

        ############## circuit output

        g2 = conv2d_same_padding(h1, self.conv_g2_w)
        if self.normalization_gate:
            self.bn_g2(g2)
        g2 = F.sigmoid(g2+self.conv_g2_b)

        if self.normalization_fgru:
            c2 = self.bn_c2(h1)
        c2 = conv2d_same_padding(c2, self.conv_c2_w)

        ############## output integration

        if self.multiplicative_excitation:
            h2_hat = F.relu( self.kappa*(h1 + c2) + self.omega*(h1 * c2) )
        else:
            h2_hat = F.relu(h1 + c2)

        h2 = (1 - g2) * prev_state2 + g2 * h2_hat
        
        return h2, h1

class SE_Attention(nn.Module):
    """ if layers > 1  downsample -> upsample """
    
    def __init__(self, 
                input_size, 
                output_size, 
                filter_size, 
                layers, 
                normalization=True, 
                normalization_type='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):
        super().__init__()
        
        if normalization_params is None:
            normalization_params={}
        
        curr_feat = input_size
        self.module_list = []
            
        for i in range(layers):
            if i == layers-1:
                next_feat = output_size
            elif i < layers//2:
                next_feat = curr_feat // 2
            else:
                next_feat = curr_feat * 2
            
            self.module_list.append(Conv2dSamePadding(curr_feat, next_feat, filter_size))
            
            if non_linearity is not None:
                nl = pt_utils.get_nl(non_linearity)
                
            if normalization is not None:
                norm = pt_utils.get_norm(normalization)(next_feat, **normalization_params)
            
            if norm_pre_nl :
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)
            
            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)
    
    def forward(self, input_):
        return self.attention(input_)

class SA_Attention(nn.Module):
    """ if layers > 1  downsample til 1 """
    
    def __init__(self, 
                input_size, 
                output_size, 
                filter_size, 
                layers, 
                normalization='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):
        super().__init__()
        
        if normalization_params is None:
            normalization_params={}
        
        curr_feat = input_size
        self.module_list = []
        for i in range(layers):
            if i == layers-1:
                next_feat = output_size
            else:
                next_feat = curr_feat // 2
            
            self.module_list.append(Conv2dSamePadding(curr_feat, next_feat, filter_size))
            
            if non_linearity is not None:
                nl = pt_utils.get_nl(non_linearity)
                
            if normalization is not None:
                norm = pt_utils.get_norm(normalization)(next_feat, **normalization_params)
            
            if norm_pre_nl :
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)

            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)
    
    def forward(self, input_):
        return self.attention(input_)

class GALA_Attention(nn.Module):
    """ if layers > 1  downsample til spatial saliency is 1 """
    def __init__(self, 
                input_size, 
                output_size, 
                saliency_filter_size, 
                layers, 
                normalization='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):

        super().__init__()

        self.se = SE_Attention(input_size, output_size, 1,                    layers, 
                                normalization=normalization, # 'BatchNorm2D'
                                normalization_params=normalization_params,
                                non_linearity=non_linearity,
                                norm_pre_nl=norm_pre_nl)
        self.sa = SA_Attention(input_size, 1,           saliency_filter_size, layers,
                                normalization=normalization, # 'BatchNorm2D'
                                normalization_params=normalization_params,
                                non_linearity=non_linearity,
                                norm_pre_nl=norm_pre_nl)
    
    def forward(self, input_):
        return self.sa(input_) * self.se(input_)