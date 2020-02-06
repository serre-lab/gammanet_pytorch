
from torchvision.models.resnet import ResNet,BasicBlock
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
class ResNet18_test(ResNet):

    def __init__(self,progress=True,pretrained=True,freeze_layers=True,n_layers=7,arch='resnet18',**kwargs):
        
        super().__init__(BasicBlock,[2,2,2,2],**kwargs)
        if pretrained:
            state_dict=load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
            self.load_state_dict(state_dict)
        self.layers = [ 'conv1','bn1','maxpool',
                        'layer1','layer2','layer3','layer4']
        self.filters = [64,64,64,64,128,256,512]
        


