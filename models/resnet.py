
from torchvision.models.resnet import ResNet,BasicBlock

class ResNet18_test(ResNet):

    def __init__(self,progress=True,pretrained=True,freeze_layers=True,n_layers=7,**kwargs):
        
        super().__init__(BasicBlock,[2,2,2,2],pretrained,progress,**kwargs)
        
        self.layers = [ 'conv1','bn1','maxpool',
                        'layer1','layer2','layer3','layer4']
        self.filters = [64,64,64,64,128,256,512]
        


