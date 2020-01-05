import os

import torch
import torch.utils.data as data

from torchvision import transforms as tvt

from PIL import Image
import numpy as np

import glob

class BSDS500(data.Dataset):
    
    def __init__(self, images_path, labels_path, transform=None, normalize=[[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]], **kwargs):
        super().__init__()
        self.image_list = sorted(glob.glob(os.path.join(images_path,'*')))
        self.label_list = sorted(glob.glob(os.path.join(labels_path,'*')))
        if normalize is not None:
            self.normalize = tvt.Normalize(mean=normalize[0],std=normalize[1])
        if transform is not None:
            self.transform=transform
        else:
            self.transform = tvt.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image = Image.open(self.image_list[idx])
        label = np.load(self.label_list[idx])
        label = Image.fromarray(label)
        image = self.transform(image)
        label = self.transform(label)
        if hasattr(self,'normalize'):
            image = self.normalize(image)
        return {'image': image,'label':label}

    def get_image(self,idx):
        image = Image.open(self.image_list[idx])
        image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
        return image/255.0
    
    def get_label(self,idx):
        label = np.load(self.label_list[idx])
        return label