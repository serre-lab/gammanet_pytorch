import torch.utils.data as data
from PIL import Image
from skimage import io, transform
import os
from torchvision import transforms as tv_transforms
#from ops import transforms 
from utils import py_utils

def process_transforms(cfg):
    
    def get_transform(transform, kwargs):
        assert hasattr(tv_transforms, transform), "transform operation not found" #or hasattr(transforms, transform)

        if hasattr(tv_transforms, transform):
            return getattr(tv_transforms, transform)(**dict(kwargs))
        else:
            return getattr(tv_transforms, transform)(**dict(kwargs))
    if len(cfg)==0:
        return tv_transforms.ToTensor()
    else:
        t = [get_transform(t,cfg[t]) for t in cfg]
        if 'ToTensor' not in cfg:
            t = t+[tv_transforms.ToTensor()]
        return tv_transforms.Compose(t)


def get_dataset(cfg):
    dataset_module = py_utils.import_module(cfg.import_prepath)
    trainset, valset,testset = None,None,None 
    if 'train' in cfg:    
        trainset_class = getattr(dataset_module, cfg.train.name)
        #cfg.train.transform = process_transforms(cfg.train.transform)
        trainset = trainset_class(**dict(cfg.train,transform=process_transforms(cfg.train.transform)))
    if 'val' in cfg:
        if isinstance(cfg.val, float) and trainset is not None:
            valsize= int(cfg.val*len(trainset))
            trainset, valset = data.random_split(trainset, [len(trainset)-valsize,valsize])
        else:
            valset_class = getattr(dataset_module, cfg.val.name)
            #cfg.val.transform = process_transforms(cfg.val.transforms)
            valset = valset_class(**dict(cfg.val,transform=process_transforms(cfg.val.transform)))

    if 'test' in cfg:
        testset_class = getattr(dataset_module, cfg.test.name)
        #cfg.test.transform = process_transforms(cfg.test.transforms)
        testset = testset_class(**dict(cfg.test,transform=process_transforms(cfg.test.transform)))

    return trainset, valset, testset

def get_dataset_by_type(cfg):
    dataset_module = py_utils.import_module(cfg.import_prepath)
    
    assert ds_type in cfg, 'no %s set in config'%ds_type

    set_class = getattr(dataset_module, cfg.import_class)
    dataset = set_class(**dict(cfg,transform=process_transforms(cfg.transform)))

    return dataset

def split_dataset(dataset, sizes, split_type='random'):
    splits = []
    if split_type=='random':
        splits = data.random_split(dataset, sizes)
    
    return splits

def get_set(cfg):
    dataset_module = py_utils.import_module(cfg.import_prepath)
    
    set_class = getattr(dataset_module, cfg.import_class)
    kwargs = cfg if 'transform' not in cfg else dict(cfg,transform=process_transforms(cfg.transform))
    dataset = set_class(**kwargs)

    return dataset
