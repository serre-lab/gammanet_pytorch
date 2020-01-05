
import os
import time
import logging
import numpy as np
from collections import OrderedDict, Iterable

import torch
import torch.utils.data as ptdata
from torch.utils.tensorboard import SummaryWriter

import skimage as si
import skimage.transform

from PIL import Image

from ops import data_tools, model_tools, losses, optimizers, metrics
from utils import pt_utils, py_utils
from utils.py_utils import AverageMeter

from experiments.base import TrainExperiment

class BDTrain(TrainExperiment):

    def setup(self):
        self.setup_data()
        self.setup_dataloader()
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        self.setup_cuda()
        
    def setup_optimizer(self):
        if 'exclusion_params' in self.cfg.optim:
            param_list = self.cfg.optim.exclusion_params
            p_list =[]
            remaining_model_params = list(self.model.parameters())
            for p in param_list:
                s_p = {}
                
                if isinstance(p['params'], Iterable):
                    model_params = []
                    for mp in p['params']:
                        model_params += list(getattr(self.model,mp).parameters())
                else:
                    model_params = getattr(self.model,p['params']).parameters()
                
                remaining_model_params = list(set(remaining_model_params) - set(model_params))
                s_p['params'] = model_params
                if 'lr' in p:
                    s_p['lr'] = p['lr']
                if 'betas' in p:
                    s_p['betas'] = p['betas']
                if 'weight_decay' in p:
                    s_p['weight_decay'] = p['weight_decay']

                p_list.append(s_p)
            p_list.append({'params': remaining_model_params})
            optimizer = optimizers.get_optimizer(self.cfg.optim.name)(p_list, **self.cfg.optim.params)
            self.optimizer = optimizer

        else:
            optimizer = optimizers.get_optimizer(self.cfg.optim.name)(self.model.parameters(), **self.cfg.optim.params)
            self.optimizer = optimizer

    def load(self, path):
        super().load(path)
        if 'best_error' in checkpoint:
            self.best_error = checkpoint['best_error'] 

    def save(self, path=None):
        save_dict = {
            'model_state_dict': self.model.module.state_dict() if self.cfg.parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.cur_epoch,
            'best_error': self.best_error
        }
        torch.save(save_dict, path)

    def init_train(self):
        super().init_train()
        if 'checkpoint' not in self.cfg:
            self.best_error = 1000000
            
    def train_step(self, sample):
        start = time.perf_counter()
        
        train_input = sample[self.cfg.trainset.input]
        train_target = sample[self.cfg.trainset.target]
        
        if self.cuda:
            train_input = train_input.to(self.device)
            train_target = train_target.to(self.device)

        # print(train_input.shape)
        # print(train_input.mean())
        
        # print(train_target.shape)
        # print(train_target.mean())
        # print(train_target.max())
        output = self.model(train_input)

        # print(output.shape)
        # print(output.mean())
        # print(output.max())
        # print(output.min())
        loss = self.criterion(output, train_target)
        
        cur_score = self.accuracy(output.data, train_target)
        
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.t_hist['batch_time'].update(time.perf_counter() - start)
        self.t_hist['loss'].update(loss.data.item(), train_input.size(0))
        self.t_hist['score'].update(cur_score.data.item(), train_input.size(0))
    
    def validate(self):
        self.init_val()
        for i, sample in enumerate(self.val_loader):
            self.val_step(sample)
            if i%self.log_freq==0:
                self.log(self.get_val_log(i),t='val')
        
        self.summary(self.get_val_summary())

    def val_step(self, sample):
        start = time.perf_counter()
        
        val_input = sample[self.cfg.valset.input]
        val_target = sample[self.cfg.valset.target].float()

        if self.cuda:
            val_input = val_input.cuda()
            val_target = val_target.cuda()

        output = self.model.forward(val_input)
        
        loss = self.criterion(output, val_target)
        
        cur_score = self.accuracy(output.data, val_target)

        self.v_hist['batch_time'].update(time.perf_counter() - start)
        self.v_hist['loss'].update(loss.data.item(), val_input.size(0))
        self.v_hist['score'].update(cur_score.data.item(), val_input.size(0))   


    def run(self):
        self.init_train()
        for self.cur_epoch in range(self.cur_epoch, self.cfg.epochs+1):
            self.logger.info('train epoch %d'%(self.cur_epoch,))
            self.train_epoch()
            if self.val_set is not None and self.cur_epoch%self.cfg.val_freq==0:
                with torch.no_grad():
                    self.logger.info('validation epoch %d'%(self.cur_epoch,))
                    self.validate()
                    if self.best_error > self.v_hist['score'].avg:
                        self.best_error = self.v_hist['score'].avg
                        self.save(os.path.join(self.cfg.dir,'ckpt_%d_%.03f.pth.tar'%(self.cur_epoch,self.best_error)))
                        self.plot_recurrence(5)
                
            if self.save_freq is not None and self.cur_epoch%self.save_freq==0:
                self.save(os.path.join(self.cfg.dir,str(self.cur_epoch)+'.pth.tar'))
                with torch.no_grad():
                    self.plot_recurrence(5)
        self.save(os.path.join(self.cfg.dir,str(self.cur_epoch)+'.pth.tar'))
    
    def plot_recurrence(self,n_samples, target_shape=200):
        example_path = os.path.join(self.cfg.dir,'examples')
        py_utils.ensure_dir(example_path)

        for i in range(n_samples):
            image_idx = i*5
            sample = self.val_set[image_idx]
            im = sample[self.cfg.valset.input].cuda()
            if 'timesteps' in self.cfg.model.args:
                if self.cfg.parallel:
                    _, output = self.model.module.forward(im[None,:,:,:],return_hidden=True)
                else:
                    _, output = self.model.forward(im[None,:,:,:],return_hidden=True)
            else:
                output = self.model.forward(im[None,:,:,:])
            output = torch.sigmoid(output).cpu().numpy()[0]
            gt = sample[self.cfg.valset.target]
            im = np.transpose(im.cpu().numpy(),(1,2,0))
            
            aspect_ratio = 1.0*im.shape[1]/im.shape[0]
            out_shape = (int(target_shape),int(target_shape*aspect_ratio))
            viz = [self.val_set.get_image(image_idx)]
            if len(output.shape)>len(im.shape):
                output = np.transpose(output,(0,2,3,1))
                for o in output:
                    viz.append(o)
            else:
                output = np.transpose(output,(1,2,0))
                viz.append(output)
            viz.append(np.transpose(gt.numpy(),(1,2,0)))
            for j in range(len(viz)):
                if viz[j].shape[-1] == 1:
                    viz[j] = np.concatenate([viz[j],viz[j],viz[j]],axis=-1)
                viz[j] = si.transform.resize(viz[j],out_shape)
                viz[j] = np.pad(viz[j] ,((1,1),(1,1),(0,0)),'constant',constant_values=1)
            viz = (np.concatenate(viz,axis=1)*255).astype(np.uint8)
            
            im = Image.fromarray(viz)
            im.save(os.path.join(example_path,"%d_%i.png"%(self.cur_epoch,i)))

            
        
            
