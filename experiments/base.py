
import torch.utils.data as ptdata
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time

from collections import OrderedDict, Iterable
from ops import data_tools, model_tools, losses, optimizers, metrics
from utils import pt_utils, py_utils
from utils.py_utils import AverageMeter, load_config
import logging

import os

torch.backends.cudnn.benchmark = True

class Experiment:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.type = cfg.type
        self.logger = logging.getLogger()

    def run(self):
        pass

class TrainExperiment(Experiment):
    
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cuda = True if torch.cuda.is_available() else False

    def setup(self):
        self.setup_data()
        self.setup_dataloader()
        self.setup_model()
        self.setup_loss()
        self.setup_optimizer()
        self.setup_cuda()
        

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

    def setup_cuda(self):
        self.cuda = self.cfg.cuda and self.cuda

        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        
        if self.cfg.parallel:
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.logger.info("Loading parallel finished on GPU count: %d"%(torch.cuda.device_count(),))
        else:
            self.model = self.model.to(self.device)
        
    def setup_loss(self):
        self.criterion = losses.get_loss(self.cfg.loss)
        if 'eval' in self.cfg:
            self.accuracy = metrics.get_eval(self.cfg.eval)

    def setup_optimizer(self):
        if 'exclusion_params' in self.cfg.optim:
            param_list = self.cfg.optim.exclusion_params
            p_list =[]
            named_parameters = list(self.model.named_parameters())
            remaining_model_params = list(self.model.parameters())
            for p_vars in param_list:
                s_p = {}
                if not isinstance(p_vars['params'], str):
                    model_params = []
                    for mp in p_vars['params']:
                        n_p = [p[1] for p in named_parameters if mp in p[0]]
                        model_params += n_p #list(getattr(self.model,mp).parameters())
                        self.logger.info('%s excluded params number : %d'%(mp,len(n_p)))
                else:
                    model_params = [p[1] for p in named_parameters if p_vars['params'] in p[0]] 
                    self.logger.info('%s excluded params number : %d'%(p_vars['params'],len(model_params)))
                    #model_params = # getattr(self.model,p['params']).parameters()
                
                remaining_model_params = list(set(remaining_model_params) - set(model_params))
                
                s_p['params'] = model_params
                if 'lr' in p_vars:
                    s_p['lr'] = p_vars['lr']
                if 'betas' in p_vars:
                    s_p['betas'] = p_vars['betas']
                if 'weight_decay' in p_vars:
                    s_p['weight_decay'] = p_vars['weight_decay']

                p_list.append(s_p)

            self.logger.info('remaining params number : %d'%len(remaining_model_params))
            p_list.append({'params': remaining_model_params})
            optimizer = optimizers.get_optimizer(self.cfg.optim.name)(p_list, **self.cfg.optim.params)
            self.optimizer = optimizer

        else:
            optimizer = optimizers.get_optimizer(self.cfg.optim.name)(self.model.parameters(), **self.cfg.optim.params)
            self.optimizer = optimizer
    
    def load(self, path):
        checkpoint = torch.load(path)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # get module if model is parallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7]=='module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict

            if self.cfg.parallel:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
        # if 'optimizer_state_dict' in checkpoint:
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            self.cur_epoch = checkpoint['epoch'] 
        if 'best_score' in checkpoint:
            self.best_score = checkpoint['best_score'] 

    def load_model(self, path, load_state_dict=True):
        if load_state_dict:
            
            state_dict = torch.load(path)
            # get module if model is parallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7]=='module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict

            if self.cfg.parallel:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
        else:
            self.model = torch.load(path)
            self.setup_cuda()

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

    def init_train(self):
        summary_path = os.path.join(self.cfg.dir,'summary')
        py_utils.ensure_dir(summary_path)
        self.writer = SummaryWriter(log_dir=summary_path)
        self.num_epochs = self.cfg.epochs
        self.n_train_batches = len(self.train_loader)
        self.n_val_batches = len(self.val_loader)
        self.global_iters = 0
        
        # sample = next(iter(self.train_loader))
        # sample_input = sample[self.cfg.trainset.input].to(self.device) if self.cuda else sample[self.cfg.trainset.input]
        # self.writer.add_graph(self.model,sample_input)

        self.save_freq = self.cfg.save_freq if 'save_freq' in self.cfg else None
        self.log_freq = self.cfg.log_freq
        self.sum_freq = self.cfg.sum_freq if 'sum_freq' in self.cfg else None

        if 'checkpoint' in self.cfg:
            self.load(self.cfg.checkpoint)
        else:
            self.best_score = 0
            self.cur_epoch = 1
        
    def init_train_epoch(self):
        self.model.train()
        self.t_hist = {}
        self.t_hist['batch_time'] = AverageMeter()
        self.t_hist['loss'] = AverageMeter()
        self.t_hist['score'] = AverageMeter()

    def init_val(self):
        self.model.eval()
        self.v_hist = {}
        self.v_hist['batch_time'] = AverageMeter()
        self.v_hist['loss'] = AverageMeter()
        self.v_hist['score'] = AverageMeter()
        
    def train_epoch(self):
        self.init_train_epoch()
        for i, sample in enumerate(self.train_loader):
            self.train_step(sample)
            if i%self.log_freq==0:
                self.log(self.get_train_log(i))
            if self.global_iters%self.sum_freq == 0:
                self.summary(self.get_train_summary())
            self.global_iters += 1
            
    def train_step(self, sample):
        start = time.perf_counter()
        
        train_input = sample[self.cfg.trainset.input]
        train_target = sample[self.cfg.trainset.target].float()
        
        if self.cuda:
            train_input = train_input.to(self.device)
            train_target = train_target.to(self.device)

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
    
    def track_grad_stats(self):
        if isinstance(self.cfg.track_grads.grad_stats.params, str):
            params = [self.cfg.track_grads.grad_stats.params]
        else:
            params = self.cfg.track_grads.grad_stats.params
        
        train_summary = []
        
        for p in self.model.named_parameters():
            if p[1].requires_grad and any(param in p[0] for param in params):
                g = p[1].grad.cpu().data.norm(2).item() / np.sqrt(np.prod(list(p[1].grad.shape)))
                #print(g)
                train_summary.append(('scalar', 'grads/'+p[0], g))
        self.summary(train_summary)
        
    def track_grad_hist(self):
        if isinstance(self.cfg.track_grads.grad_hist.params, str):
            params = [self.cfg.track_grads.grad_hist.params]
        else:
            params = self.cfg.track_grads.grad_hist.params
        
        train_summary = []

        for p in self.model.named_parameters():
            if p[1].requires_grad and any(param in p[0] for param in params):
                g = p[1].grad.cpu().data.numpy().flatten()
                if len(g) > 500:
                    g = g[np.random.choice(len(g), 500, replace=False)]
                #print(g.shape)
                train_summary.append(('histogram', 'grad_hist/'+p[0], g))
        self.summary(train_summary)

    def track_grad_flow(self):
        
        n_p = self.model.module.named_parameters() if hasattr(self.model,'module') else self.model.named_parameters()

        fig = pt_utils.plot_grad_flow_v2(n_p)

        train_summary = [('figure', 'grad_flow/grad_flow', fig)]

        self.summary(train_summary)
        
    def track_weight_hist(self):
        if isinstance(self.cfg.track_weights.params, str):
            params = [self.cfg.track_weights.params]
        else:
            params = self.cfg.track_weights.params
        
        train_summary = []

        for p in self.model.named_parameters():
            if p[1].requires_grad and any(param in p[0] for param in params):
                g = p[1].cpu().data.numpy().flatten()
                if len(g) > 1000:
                    g = g[np.random.choice(len(g), 1000, replace=False)]
                train_summary.append(('histogram', 'weights/'+p[0], g))
        self.summary(train_summary)
    
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
        
        cur_score = self.accuracy(output.data, **sample)

        self.v_hist['batch_time'].update(time.perf_counter() - start)
        self.v_hist['loss'].update(loss.data.item(), val_input.size(0))
        self.v_hist['score'].update(cur_score.data.item(), val_input.size(0))

    def get_val_summary(self):
        val_summary = []
        val_summary.append(('scalar', 'loss/val', np.mean(self.v_hist['loss'].avg)))
        val_summary.append(('scalar', 'score/val', np.mean(self.v_hist['score'].avg)))
        return val_summary
    
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

    def log(self,kwargs,t='train'):
        out = ''
        if 'epoch' in kwargs:
            out += '[epoch {0:0{2}d}/{1}]'.format(kwargs.pop('epoch'),self.num_epochs,len(str(self.num_epochs)))
        if 'batch' in kwargs:
            n_batch = self.n_train_batches if t=='train' else self.n_val_batches
            out += '[b {0:0{2}d}/{1}]'.format(kwargs.pop('batch'),n_batch,len(str(n_batch)))
        for k,v in kwargs.items():
            if isinstance(v,int):
                str_v = '{0:05d}'.format(v)
            elif isinstance(v,float):
                str_v = '%.04f'%v
            else:
                str_v = v
            out += '[{0} {1}]'.format(k,str_v)

        self.logger.info(out)

    def summary(self,s_list):
        for s_type, s_title, s_value in s_list:
            getattr(self.writer, 'add_'+s_type)(s_title, s_value, self.global_iters)

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
        
            
class TestExperiment(Experiment):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cuda = True if torch.cuda.is_available() else False

        self.exp_cfg = load_config(self.cfg.exp_cfg)
    
    def setup(self):
        self.setup_data()
        self.setup_dataloader()
        self.setup_model()
        self.setup_loss()
        self.setup_cuda()

    def setup_data(self):
        self.test_set = data_tools.get_set(self.cfg.testset)
        
    def setup_dataloader(self):
        self.test_loader = ptdata.DataLoader(self.test_set, **self.cfg.dataloader.kwargs)

    def setup_model(self):
        self.model = model_tools.get_model(self.exp_cfg.model)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('number of trainable parameters : %d'%params)

    def setup_loss(self):
        if not 'loss' in self.cfg:
            self.cfg['loss'] = self.exp_cfg.loss
        
        self.criterion = losses.get_loss(self.cfg.loss)
        if 'eval' in self.cfg or 'eval' in self.exp_cfg:
            if not 'loss' in self.cfg:
                self.cfg['eval'] = self.exp_cfg.eval
            self.accuracy = metrics.get_eval(self.cfg.eval)

    def setup_cuda(self):
        self.cuda = self.cfg.cuda and self.cuda
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model = self.model.to(self.device)
    
    def load(self, path):
        checkpoint = torch.load(path)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # get module if model is parallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7]=='module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict

            if self.cfg.parallel:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
                
        if 'epoch' in checkpoint:
            self.test_epoch = checkpoint['epoch'] 

    def load_model(self, path, load_state_dict=True):
        if load_state_dict:
            
            state_dict = torch.load(path)
            # get module if model is parallel
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7]=='module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            state_dict = new_state_dict

            if self.cfg.parallel:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            
        else:
            self.model = torch.load(path)
            self.setup_cuda()

    def init_test(self):
        
        self.model.eval()
        self.t_hist = {}
        self.t_hist['loss'] = []
        self.t_hist['score'] = []

    def test_step(self, sample):

        test_input = sample[self.cfg.testset.input]
        test_target = sample[self.cfg.testset.target].float()

        if self.cuda:
            test_input = test_input.to(self.device)
            test_target = test_target.to(self.device)

        output = self.model.forward(test_input)
        
        loss = self.criterion(output, test_target)
        
        cur_score = self.accuracy(output.data, **sample)

        self.t_hist['loss'].append(loss.data.item())
        self.t_hist['score'].append(cur_score.data.item())
        self.t_hist['logits'].append(output)

    def test(self):
        self.init_test()
        for i, sample in enumerate(self.test_loader):
            self.test_step(sample)
        self.logger.info('mean loss ', np.mean(self.t_hist['loss']))
        self.logger.info('mean score ', np.mean(self.t_hist['score']))

    def run(self):
        self.load(self.cfg.checkpoint)
        self.test()
        np.save(os.path.join(self.cfg.dir, 'results.npy'), self.t_hist)
        
class CrossValExperiment(Experiment):
    """
    Runs many experiments with different folds.
    1- setup folder and experiment configs
    2- let user run each config independently
    3- combine validation results to get a final score
    """
    def __init__(self,cfg):
        super().__init__(cfg)