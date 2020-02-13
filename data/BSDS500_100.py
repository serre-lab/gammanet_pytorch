import os

import torch
import torch.utils.data as data

from torchvision import transforms as tvt

from PIL import Image
import numpy as np

import glob

class BSDS500(data.Dataset):
    
    def __init__(self, images_path, labels_path, resize=None, normalize=[[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]], **kwargs):
        super().__init__()
        self.image_list = sorted(glob.glob(os.path.join(images_path,'*')))
        self.label_list = sorted(glob.glob(os.path.join(labels_path,'*')))
        if normalize is not None:
            self.normalize = tvt.Normalize(mean=normalize[0],std=normalize[1])
        if resize is not None:
            resize = [resize,resize] if isinstance(resize,int) else resize
            print(resize)
            self.transform = tvt.Compose([tvt.Resize(size=resize), tvt.ToTensor()])
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
        label = label.max(0, keepdims=True).values
        # print(label)
        label = label/label.max()
        if hasattr(self,'normalize'):
            image = self.normalize(image)
        return {'image': image,'label':label}

    def get_image(self,idx):
        image = Image.open(self.image_list[idx])
        image = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
        return image/255.0
    
    def get_label(self,idx):
        label = np.load(self.label_list[idx])
        return label
    
    def get_name(self,idx):
        return self.image_list[idx].split('/')[-1].split('.')[0]


class BSDS500_AUG(data.Dataset):
    
    def __init__(self, data_path, list_file, resize=None, normalize=[[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]], **kwargs):
        super().__init__()
        # self.image_list = sorted(glob.glob(os.path.join(images_path,'*')))
        with open(list_file, 'r') as f:
            image_list = f.readlines()
        self.image_list = [[a.split(' ')[0], a.split(' ')[1].replace('\n','')] for a in image_list]
        self.data_path = data_path
        if normalize is not None:
            self.normalize = tvt.Normalize(mean=normalize[0],std=normalize[1])
        if resize is not None:

            resize = [resize,resize] if isinstance(resize,int) else resize
            print(resize)
            self.transform = tvt.Compose([tvt.Resize(size=resize), tvt.ToTensor()])
        else:
            self.transform = tvt.ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image = Image.open(os.path.join(self.data_path,self.image_list[idx][0]))
        # label = np.load(self.label_list[idx][1])
        label = Image.open(os.path.join(self.data_path,self.image_list[idx][1]))
        image = self.transform(image)
        label = self.transform(label)
        label = label.max(0, keepdims=True).values
        label /= label.max()
        if hasattr(self,'normalize'):
            image = self.normalize(image)
        return {'image': image,'label':label}

    def get_image(self,idx):
        image = Image.open(os.path.join(self.data_path,self.image_list[idx][0]))
        image = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
        return image/255.0
    
    def get_label(self,idx):
        # label = np.load(self.label_list[idx])
        label = Image.open(os.path.join(self.data_path,self.image_list[idx][1]))
        label = np.array(label.getdata()).reshape(label.size[1], label.size[0], 3).mean(-1, keepdims=True)
        return label
    
    def get_name(self,idx):
        return self.image_list[idx][0].split('/')[-1].split('.')[0]