import torchvision
import torch
from collections import OrderedDict

keys_mappings = {'features.0.weight': 'conv1_1.weight', 
                'features.0.bias': 'conv1_1.bias', 
                'features.2.weight': 'conv1_2.weight', 
                'features.2.bias': 'conv1_2.bias', 
                'features.5.weight': 'conv2_1.weight', 
                'features.5.bias': 'conv2_1.bias', 
                'features.7.weight': 'conv2_2.weight', 
                'features.7.bias': 'conv2_2.bias', 
                'features.10.weight': 'conv3_1.weight', 
                'features.10.bias': 'conv3_1.bias', 
                'features.12.weight': 'conv3_2.weight', 
                'features.12.bias': 'conv3_2.bias', 
                'features.14.weight': 'conv3_3.weight', 
                'features.14.bias': 'conv3_3.bias', 
                'features.17.weight': 'conv4_1.weight', 
                'features.17.bias': 'conv4_1.bias', 
                'features.19.weight': 'conv4_2.weight', 
                'features.19.bias': 'conv4_2.bias', 
                'features.21.weight': 'conv4_3.weight', 
                'features.21.bias': 'conv4_3.bias', 
                'features.24.weight': 'conv5_1.weight', 
                'features.24.bias': 'conv5_1.bias', 
                'features.26.weight': 'conv5_2.weight', 
                'features.26.bias': 'conv5_2.bias', 
                'features.28.weight': 'conv5_3.weight', 
                'features.28.bias': 'conv5_3.bias', 
                }


model = torchvision.models.vgg16(pretrained=True)
state_dict = model.state_dict()

new_state_dict = OrderedDict()

for k,v in keys_mappings.items():
    new_state_dict[v] = state_dict[k]

torch.save(new_state_dict, 'model_weights/vgg_16.pth.tar')

