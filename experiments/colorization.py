
import torch.utils.data as ptdata
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time

from collections import OrderedDict, Iterable
from ops import data_tools, model_tools, losses, optimizers, metrics
from utils import pt_utils, py_utils
from utils.py_utils import AverageMeter
import logging

import os

from colorization_utils import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer

class Experiment:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.type = cfg.type
        self.logger = logging.getLogger()

    def run(self):
        pass

class TrainExperiment(Experiment):
       
    def setup_data(self):
        train_set = data_tools.get_set(self.cfg.trainset)
        
        if 'valset' in self.cfg:
            if isinstance(self.cfg.valset, float) and trainset is not None:
                valsize= int(self.cfg.valset*len(train_set))
                train_set, val_set = data.random_split(train_set, [len(train_set)-valsize,valsize])
            else:
                val_set = data_tools.get_set(self.cfg.valset)
        else:
            val_set = None

        self.train_set = train_set
        self.val_set = val_set
        
    def setup_dataloader(self):
        self.train_loader = ptdata.DataLoader(self.train_set, **self.cfg.dataloader.kwargs)
        self.val_loader = ptdata.DataLoader(self.val_set, **self.cfg.dataloader.kwargs) if self.val_set is not None else None
    
    def setup_model(self):
        self.model = model_tools.get_model(self.cfg.model)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('number of trainable parameters : %d'%params)

        self.encode_layer = NNEncLayer()
        self.boost_layer = PriorBoostLayer()
        self.nongray_mask = NonGrayMaskLayer()
        
    def setup_loss(self):
        self.criterion = losses.get_loss(self.cfg.loss)
        if 'eval' in self.cfg:
            self.accuracy = metrics.get_eval(self.cfg.eval)

    def save(self, path=None):
        save_dict = {
            'model_state_dict': self.model.module.state_dict() if self.cfg.parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.cur_epoch,
            'best_score': self.best_score
        }
        torch.save(save_dict, path)

    def save_model(self, path):
        if self.cfg.parallel:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
            
    def train_step(self, sample):
        start = time.perf_counter()
        
        train_input = sample[self.cfg.trainset.input]
        train_target = sample[self.cfg.trainset.target].float()
        
        if self.cuda:
            train_input = train_input.to(self.device)
            train_target = train_target.to(self.device)

        images = images.unsqueeze(1).float().cuda()
        img_ab = img_ab.float()
        encode,max_encode=encode_layer.forward(img_ab)
        targets=torch.Tensor(max_encode).long().cuda()
        boost=torch.Tensor(boost_layer.forward(encode)).float().cuda()
        mask=torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()
        boost_nongray=boost*mask
        outputs = model(images)#.log()
        output=outputs[0].cpu().data.numpy()
        out_max=np.argmax(output,axis=0)

        print('set',set(out_max.flatten()))
        loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
        
        output = self.model(train_input)

        loss = self.criterion(output, train_target)
        
        cur_score = self.accuracy(output.data, **sample)
        
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.t_hist['batch_time'].update(time.perf_counter() - start)
        self.t_hist['loss'].update(loss.data.item(), train_input.size(0))
        self.t_hist['score'].update(cur_score.data.item(), train_input.size(0))

    def get_train_summary(self):
        train_summary = []
        train_summary.append(('scalar', 'loss/train', np.mean(self.t_hist['loss'].history[-self.log_freq:])))
        train_summary.append(('scalar', 'score/train', np.mean(self.t_hist['score'].history[-self.log_freq:])))
        return train_summary
       
    def get_train_log(self,i):
        log_dict = OrderedDict()
        
        log_dict['epoch'] = self.cur_epoch 
        log_dict['batch'] = i
        log_dict['b_t'] = np.mean(self.t_hist['batch_time'].history[-self.log_freq:])
        
        log_dict['L'] = self.t_hist['loss'].val
        log_dict['L_m'] = np.mean(self.t_hist['loss'].history[-self.log_freq:])
        log_dict['L_em'] = self.t_hist['loss'].avg
        log_dict['S'] = self.t_hist['score'].val
        log_dict['S_m'] = np.mean(self.t_hist['score'].history[-self.log_freq:])
        log_dict['S_em'] = self.t_hist['score'].avg
        
        return log_dict

    def val_step(self, sample):
        start = time.perf_counter()
        
        val_input = sample[self.cfg.valset.input]
        val_target = sample[self.cfg.valset.target].float()

        if self.cuda:
            val_input = val_input.cuda()
            val_target = val_target.cuda()

        output = self.model.forward(val_input)
        
        loss = self.criterion(output, val_target)
        
        cur_score = self.accuracy(output.data, **sample)

        self.v_hist['batch_time'].update(time.perf_counter() - start)
        self.v_hist['loss'].update(loss.data.item(), val_input.size(0))
        self.v_hist['score'].update(cur_score.data.item(), val_input.size(0))
    
    def get_val_log(self,i):
        log_dict = OrderedDict()
        log_dict['epoch'] = self.cur_epoch 
        log_dict['batch'] = i
        log_dict['b_t'] = np.mean(self.v_hist['batch_time'].history[-self.log_freq:])
        
        log_dict['L'] = self.v_hist['loss'].val
        log_dict['L_m'] = np.mean(self.v_hist['loss'].history[-self.log_freq:])
        log_dict['L_em'] = self.v_hist['loss'].avg
        log_dict['S'] = self.v_hist['score'].val
        log_dict['S_m'] = np.mean(self.v_hist['score'].history[-self.log_freq:])
        log_dict['S_em'] = self.v_hist['score'].avg
        
        return log_dict

    def run(self):
        self.init_train()
        for self.cur_epoch in range(self.cur_epoch, self.cfg.epochs+1):
            self.logger.info('train epoch %d'%(self.cur_epoch,))
            self.train_epoch()
            
            if self.val_set is not None:
                self.logger.info('validation epoch %d'%(self.cur_epoch,))
                self.validate()
                if self.best_score < self.v_hist['score'].avg:
                    self.best_score = self.v_hist['score'].avg
                    self.save(os.path.join(self.cfg.dir,'ckpt_%d_%.03f.pth.tar'%(self.cur_epoch,self.best_score)))
            
            if self.save_freq is not None and self.cur_epoch%self.save_freq==0:
                self.save(os.path.join(self.cfg.dir,str(self.cur_epoch)+'.pth.tar'))
        self.save(os.path.join(self.cfg.dir,str(self.cur_epoch)+'.pth.tar'))
        