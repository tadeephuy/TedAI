import os
import collections
import gc
import torch
from functools import partial
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, HTML, clear_output
from fastprogress.fastprogress import master_bar, progress_bar

from tedai.utils import *
################ DATASET ################
class TedImageDataset(Dataset):
    def __init__(self, data_path, df, transforms=None, img_size=224, label_cols_list=None):
        self.data_path,self.img_size,self.df = data_path,img_size,df
        self.transforms = transforms or self.default_transforms()
        self.transforms = partial(self.transforms, img_size=self.img_size)()
        
        # labels
        self.label_cols_list = label_cols_list
        if self.label_cols_list is not None:
            self.labels = self.df[self.label_cols_list].to_numpy()
        
    def __getitem__(self, idx):
        img = self._imread(self.df.Images[idx])
        if self.label_cols_list is not None:
            label = self.labels[idx]
            return self.transforms(img), label.astype(float)
        return self.transforms(img)

    def __len__(self): return len(self.df)

    def _imread(self, img_path): 
        return cv2.cvtColor(cv2.imread(os.path.join(self.data_path, img_path)), cv2.COLOR_BGR2RGB)

    @staticmethod
    def default_transforms():
        return lambda img_size: Compose([ToPILImage(), Resize(int(img_size*1.3)), CenterCrop((img_size, img_size)), ToTensor(), 
                                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
################ NETWORK ################
class BasicHead(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(BasicHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.BatchNorm1d(in_features*2), nn.Dropout(0.25), 
                                nn.Linear(in_features*2, in_features), nn.LeakyReLU(),
                                nn.BatchNorm1d(in_features), nn.Dropout(0.5), 
                                nn.Linear(in_features, hidden_size))

    def forward(self, x):
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)
        x = self.flatten(x)
        return self.fc(x)

class TedModel(nn.Module):
    """
    class wrapper for basic architecture used with `TedLearner`
    """
    def __init__(self, arch, hidden_size, num_classes, head=None):
        super(TedModel, self).__init__()
        self.base = arch
        self.head = self.create_head(hidden_size, num_classes) if head is None else head
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight)

    def forward(self, x): return self.head(self.base(x))

    @staticmethod
    def create_head(in_features, hidden_size): return BasicHead(in_features, hidden_size)
        
################ DATA ################
class TedData:
    """
    Fastai databunch immitation
    """
    def __init__(self, data_path, ds_class, transforms, img_size=224, bs=32, n_workers=8, fix_dis=False):
        """
        Args:
            data_path: str - path to the data
            ds_class: tuple | class - 
                      (train_dataset_class, val_dataset_class) or 
                      (train_dataset_class, val_dataset_class, test_dataset_class)
                      can pass train_dataset_class ->  (train_dataset_class, train_dataset_class) 
            transforms: tuple | class - (train_transforms, val_transforms) can pass train_transforms -> (train_transforms, None)
        """
        self.data_path = data_path
        self.img_size, self.bs, self.fix_dis = img_size, bs, fix_dis
        self.n_workers = n_workers

        if not isinstance(ds_class, tuple): 
            self.ds_class = (ds_class, ds_class)
        else: self.ds_class = ds_class

        if isinstance(transforms, tuple): 
            self.transforms = transforms
        else: self.transforms = (transforms, None)

        self._initialize_data()

    def set_size(self, img_size, bs=None, n_workers=None):
        """
        function to set img size for the Data
        """
        self.img_size = img_size
        self.bs = bs or self.bs
        self.n_workers = n_workers or self.n_workers
        self._initialize_data()

    def _initialize_data(self):
        torch.cuda.empty_cache()
        gc.collect()

        self.train_ds = self._create_ds(self.ds_class[0], transforms=self.transforms[0], img_size=self.img_size)
        self.val_ds = self._create_ds(self.ds_class[1], transforms=self.transforms[1], img_size=self.img_size, fix_dis=self.fix_dis)
        self.train_dl = self._create_dl(self.train_ds, shuffle=True)
        self.val_dl = self._create_dl(self.val_ds, shuffle=False)
        if len(self.ds_class) == 3:
            self.add_test(test_ds=self.ds_class[2])
        else:
            self.test_dl = None

    def _create_ds(self, ds_class, transforms=None, img_size=None, fix_dis=False, **kwargs):
        if fix_dis: 
            img_size = int(1.4*img_size)
        return ds_class(data_path=self.data_path, transforms=transforms, img_size=img_size, **kwargs)

    def _create_dl(self, dataset, shuffle, bs=None, **kwargs):
        return DataLoader(dataset=dataset, batch_size=bs or self.bs, shuffle=shuffle, 
                          num_workers=self.n_workers, pin_memory=True, **kwargs)
    
    def add_test(self, test_ds):
        self.test_ds = self._create_ds(test_ds, transforms=self.transforms[1], img_size=self.img_size, fix_dis=self.fix_dis)
        self.test_dl = self._create_dl(self.test_ds, shuffle=False)

    def show_batch(self, n_row=8, n_col=1, mode='train'):
        """
        function to show images batch that is fed for training
        
        Args:
            n_row: number of images in a row
            n_col: number of images in a column
            mode: `train` or `valid` to specify the img from each session
        """
        n_row = self.bs if n_row >= self.bs else n_row
        dl = {
            'train': self.train_dl, 
            'valid': self.val_dl, 
            'test': self.test_dl or self.val_dl
        }.get(mode, self.train_dl)
        
        it = iter(dl)
        for _ in range(n_col):
            xb, _ = it.next()
            make_imgs(xb, n_row=n_row, plot=True)

################ RECORDER ################
class TedRecorder:
    """
    Fastai recorder immitation
    """
    def __init__(self, learner, metrics, log='log.html'):
        self.learner = learner
        self.log = log
        # intialize metrics
        if isinstance(metrics, list): self.metrics = {metric.__name__: [] for metric in metrics}
        elif metrics is not None: self.metrics = {metrics.__name__: []}
        else: self.metrics = []

        # save model
        self.model_save = False

        self.reset()

    def reset(self, n_epochs=None):
        self.train_loss = []
        self.val_loss = []
        for k in self.metrics:
            self.metrics[k] = []
        self.lr = []
        self.step = 0
        self.epoch = 0

        # track result by epochs
        self.n_epochs = n_epochs
        self.epoch_report = pd.DataFrame(
            columns=['step/epoch', 'train_loss', 'val_loss', *[mtr for mtr in (self.metrics or [])]])
        self.master_bar = None
        
    def plot_losses(self):
        """ Plot train loss and valid loss
        """
        fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111)
        ax.yaxis.set_ticks(np.arange(-0.5, 2, 0.1))
        x_train_loss = [c[0] for c in self.train_loss]
        y_train_loss = [c[1] for c in self.train_loss]
        plt.plot(x_train_loss, y_train_loss, label='train')
        x_val_loss = [c[0] for c in self.val_loss]
        y_val_loss = [c[1] for c in self.val_loss]
        plt.plot(x_val_loss,y_val_loss, label='valid')
        plt.xlabel("steps")
        plt.ylabel("Loss")
        ax.legend()
        ax.grid(True, 'both', linestyle='dotted')
        plt.title('Train Loss & Validation Loss', {'fontsize': 20})

    def plot_metrics(self):
        """ Plot metrics value
        """
        fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111)
        ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
        plt.xlabel("steps")
        plt.ylabel("value")
        for mtrc, values in self.metrics.items():
            x_mtrc = [c[0] for c in values]
            y_mtrc = [c[1] for c in values]
            plt.plot(x_mtrc, y_mtrc, label=mtrc)
        ax.legend()
        ax.grid(True, 'major', linestyle='dotted')
        plt.title('Metrics', {'fontsize': 20})

    def plot_lr(self, layer='head'):
        """ Plot learning rate during training, layer can be either [`head`, 'base', `all`]
        """
        plt.figure(figsize=(10,7))
        plt.xlabel("steps")
        plt.ylabel("lr")
        x_lr = [c[0] for c in self.lr]
        if layer=='head' or layer=='all':
            y_lr = [c[1][-1] for c in self.lr]
            plt.plot(x_lr, y_lr, label='head')
            plt.legend()
        if layer=='base' or layer=='all': 
            y_lr = [c[1][0] for c in self.lr]
            plt.plot(x_lr, y_lr, label='base')
            plt.legend()
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0.01, 10)) 
        plt.grid(True, linestyle='dotted')
        plt.title('Learning Rate', {'fontsize': 20})
    
    def config_model_save(self, name='model', mode='reduce', monitor='val_loss', cycle_len=None):
        """
        save model function that with 4 modes: [improve, reduce, every_step, every_epoch]
        
        `improve`     : save model by checking the monitored value improvement after each validation
        `reduce`      : save model by checking the monitored value reduction after each validation
        `every_step`  : save model every `cycle_len` validation step
        `every_epoch` : save model every `cycle_len` epoch
        """
        if mode not in ['reduce', 'improve', 'every_step', 'every_epoch']:
            self.model_save = False
            print(f'No save setting mode found for {mode}, please choose `reduce`, `improve`, `every_epoch` or `every_step`')
            return
        
        self.model_save = True
        self.model_save_mode = mode
        self.model_save_name = name

        if mode in ['reduce', 'improve']:
            self.model_save_monitor = monitor or 'val_loss'
            print(f'Save model by monitoring the {self.model_save_mode} of {self.model_save_monitor} after each validation step')
        elif mode == 'every_step':
            self.model_save_cycle_len = cycle_len or 1000
            print(f'Save model every {self.model_save_cycle_len} validation step')
        elif mode == 'every_epoch':
            self.model_save_cycle_len = cycle_len or 1
            print(f'Save model every {self.model_save_cycle_len} epoch')
    
    def _model_save(self):
        save_funcs = {
            'reduce': __model_save_reduce,
            'improve': __model_save_improve,
            'every_step': __model_save_step,
            'every_epoch': __model_save_epoch,
        }
        save_funcs[self.model_save_mode]()
        

    def __model_save_reduce(self):
        monitor_value = self.epoch_report.at[-1, self.model_save_monitor]
        if monitor_value > self.old_monitor_value:
            return None
        self.old_monitor_value = monitor_value
        name = f'{self.model_save_name}-best-{self.model_save_monitor}'
        self.learner.save(name)
            
    def __model_save_improve(self):
        monitor_value = self.epoch_report.at[-1, self.model_save_monitor]
        if monitor_value < self.old_monitor_value:
            return None
        self.old_monitor_value = monitor_value
        name = f'{self.model_save_name}-best-{self.model_save_monitor}'
        self.learner.save(name)

    def __model_save_step(self):
        pass
    def __model_save_epoch(self):
        pass
    
    def update(self, train_loss, lr, validate_every=None):
        if validate_every is None:
            self._update(train_loss=train_loss, lr=lr, done_epoch=False); return

        if self.step % validate_every == 0 and self.step > 0:
            val_loss, metrics = self.learner.validate()
            self._update(train_loss=train_loss, lr=lr, val_loss=val_loss, metrics=metrics, done_epoch=False)
            self.report_epoch(done_epoch=False)
        else:
            self._update(train_loss=train_loss, lr=lr, done_epoch=False)

    def report_epoch(self, done_epoch=True):
        self.master_bar.write(self.str_stats, table=True)
        self.epoch_report.to_html(self.log, index=False)
        if done_epoch: self._plot_loss_epoch()

    def _update(self, train_loss=None, val_loss=None, metrics=None, lr=None, done_epoch=False):
        self._update_train_loss(train_loss)
        self._update_lr(lr)
        if done_epoch:
            self.step -= 1

        self._update_val_loss(val_loss)
        self._update_metrics(metrics)

        if done_epoch:
            self._update_epoch_report(self.train_loss[-1][-1], val_loss, metrics, mode='epoch')
            self.epoch += 1
        else:
            self._update_epoch_report(train_loss, val_loss, metrics, mode='step')
        self.step += 1

    def _update_train_loss(self, train_loss):
        if train_loss is not None:
            self.train_loss.append((self.step, train_loss))

    def _update_val_loss(self, val_loss):
        if val_loss is not None:
            self.val_loss.append((self.step, val_loss))

    def _update_metrics(self, metrics):
        if metrics is not None:
            for i, metric_name in enumerate(self.metrics.keys()):
                self.metrics[metric_name].append((self.step, metrics[metric_name]))

    def _update_lr(self, lr):
        if lr is not None:
            self.lr.append((self.step, lr))
    
    def _update_epoch_report(self, train_loss, val_loss, metrics, mode='epoch'):
        if mode == 'step' and val_loss is None: return
        index = {'epoch': f'epoch_{self.epoch}', 'step': f'{self.step}'}[mode]
        
        row = pd.DataFrame([
            [index, train_loss, val_loss, 
            *[metric_value for _, metric_value in (metrics or {}).items()]]], 
            columns=self.epoch_report.columns)
        self.epoch_report = self.epoch_report.append(row, ignore_index=True)
        self.str_stats = [index, f'{train_loss:.6f}', f'{val_loss:.6f}', *[f'{metric_value:.6f}' for _, metric_value in (metrics or {}).items()]]
        

    def _plot_loss_epoch(self):
        """ dynamically print the loss plot during the training/validation loop.
            expects epoch to start from 1.
        """
        self.epoch_report['is_epoch'] = self.epoch_report['step/epoch'].apply(lambda x: 'epoch' in x)
        filtered_report = self.epoch_report[self.epoch_report['is_epoch'] == True]
        
        train_loss = filtered_report.train_loss.to_numpy()
        val_loss = filtered_report.val_loss.to_numpy()
        
        self.epoch_report.drop('is_epoch', axis=1, inplace=True)
        
        x = range(self.epoch)
        y = np.concatenate((train_loss, val_loss))
        graphs = [[x,train_loss], [x,val_loss]]
        x_margin = 0.2
        y_margin = 0.05
        x_bounds = [0-x_margin, self.n_epochs+x_margin-1]
        y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

        self.master_bar.update_graph(graphs, x_bounds, y_bounds)

    def show_imgs(self, imgs):
        if not hasattr(self.master_bar, 'imgs_out'):
            self.master_bar.imgs_out = display(imgs, display_id=True)
        else: 
            self.master_bar.imgs_out.update(imgs)
    
    def show_report(self, mode='all'):
        """ show training report, mode can be either [`all`, `epoch`, `step`]
        """
        mode = 'all' if mode not in ['all', 'epoch', 'step'] else mode
        
        df = self.epoch_report.copy(deep=True)
        if mode == 'all':
            display(HTML(df.to_html(index=False))); return
        
        if mode in ['epoch', 'step']:
            df['mode'] = df['step/epoch'].apply(lambda x: 'epoch' if 'epoch' in x else 'step')
            df = df[df['mode'] == mode].drop('mode', axis=1)
            display(HTML(df.to_html(index=False))); return
        
        

################ LEARNER ################
class TedLearner:
    """
    Fastai learner immitation
    """
    def __init__(self, data, model, loss_func, opt_func=optim.AdamW, 
                 metrics=None, device=torch.device('cuda:0'), model_path='', validate_every=None, 
                 lr_div_factor=25, clip=0.0, log='log.html', show_imgs=False):
        self.data,self.model,self.loss_func,self.metrics,self.device=data,model,loss_func,metrics,device
        self.model_path,self.log = model_path,log
        self.show_imgs = show_imgs
        self.opt_func = opt_func
        self.opt = self.opt_func([{'params': self.model.base.parameters(), 'lr': 1e-4},
                                 {'params': self.model.head.parameters(), 'lr': 1e-3}])

        self.recorder = TedRecorder(learner=self, metrics=metrics, log=self.log)
        self.lr_div_factor,self.clip = lr_div_factor,clip
        self.master_bar = None
        self.validate_every = validate_every if isinstance(validate_every, int) else None
        self.to(self.device)
    
    def _reset_state(self, n_epochs):
        """
        reset optimizer states
        reset recorder states
        reset/create master progress bar using n_epochs
        set master_bar atribute to recorder
        """
        self.opt = self.opt_func([{'params': self.model.base.parameters(), 'lr': 1e-4},
                                  {'params': self.model.head.parameters(), 'lr': 1e-3}])
        self.recorder.reset(n_epochs)
        self.master_bar = master_bar(range(n_epochs))
        self.recorder.master_bar = self.master_bar
        str_stats = list(self.recorder.epoch_report.columns)
        self.master_bar.write(str_stats, table=True)

    def to(self, device):
        self.model = self.model.to(device)
        self.loss_func = self.loss_func.to(device)
    
    def validate(self, mode='valid'):
        self.model.eval()
        val_loss_value = AverageMeter()
        preds, targets, metrics = [], [], []
        dl = {
            'train': self.data.train_dl,
            'valid': self.data.val_dl,
            'test': self.data.test_dl or self.data.val_dl
        }.get(mode, self.data.val_dl)
        with torch.no_grad():
            for xb, yb in progress_bar(dl, parent=self.master_bar):
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss = self.loss_func(out, yb).to(self.device)
                preds.append(out.cpu().numpy()), targets.append(yb.cpu().numpy())
                val_loss_value.update(val=loss.item())
        self.model.train()
        preds, targets = np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)
        if self.metrics is not None: 
            metrics = {metric.__name__: metric(preds, targets)
                       for metric in self.metrics}
        else: metrics = None
        return val_loss_value.avg, metrics

    def fit(self, n_epochs, lr=None, lr_scheduler=None, name='model', **kwargs):
        self._reset_state(n_epochs)
        if lr is not None: self._set_lr(lr)
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler(self.opt, **kwargs)
        else:
            last_lr = None
        
        # train
        ## on train begin
        train_loss_value = AverageMeter()
        for epoch in self.master_bar:
        ## on epoch begin
            self.model.train()
            self.train_bar = progress_bar(self.data.train_dl, parent=self.master_bar)
            for j, (xb, yb) in enumerate(self.train_bar):
                ## on batch begin
                xb, yb, = xb.to(self.device), yb.to(self.device)
                ## on loss begin
                loss = self.loss_func(self.model(xb), yb).to(self.device)
                ## on backward begin
                loss.backward()
                ## on backward end
                if self.clip: nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                ## on step begin
                self.opt.step()
                self.opt.zero_grad()

                if lr_scheduler is not None:
                    last_lr = lr_scheduler.get_last_lr()
                    lr_scheduler.step()
                ## on step end
                train_loss_value.update(val=loss.item())
                train_loss = train_loss_value.avg
                self.recorder.update(train_loss=train_loss, lr=last_lr, validate_every=self.validate_every)
                ## on batch end

            # validation
            ## on validation begin
            val_loss, metrics = self.validate()
            ## on validation end
            self.recorder._update(val_loss=val_loss, metrics=metrics, done_epoch=True)
            self.recorder.report_epoch(done_epoch=True)
            # show training images
            if self.show_imgs == True: self.recorder.show_imgs(make_imgs(xb.detach(), plot=False))

        ## on train end
        # save model and optimizer states
        self.save(name=name)
    
    def get_preds(self, mode='test', with_losses=False, activ=None):
        """
        get all predictions of a given dataset by specifying mode `train`, `valid` or `test`
        with_losses:
            True: the dataset has labels, get the losses and label -> (preds, labels, losses)
            False: only get predictions -> preds
        """
        self.model.eval()
        preds, targets, losses = [], [], []
        ds = {
            'train': self.data.train_ds,
            'valid': self.data.val_ds,
            'test': self.data.test_ds or self.data.val_ds
        }.get(mode, self.data.val_ds)
        dl = self.data._create_dl(dataset=ds, shuffle=False, bs=None if not with_losses else 1)
        with torch.no_grad():
            for batch in progress_bar(dl):
                xb = batch[0].to(self.device)
                out = self.model(xb)
                if activ is not None: out = activ(out)
                preds.append(out.cpu().numpy())
                if len(batch) == 2 and with_losses:
                    yb = batch[1].to(self.device)
                    loss = self.loss_func(out, yb).cpu().numpy()
                    target = yb.cpu().numpy()
                    targets.append(target)
                    losses.append(loss)
        preds = np.concatenate(preds, axis=0)
        if len(batch) == 2 and with_losses:
            targets = np.concatenate(targets, axis=0)
            losses = np.expand_dims(np.array(losses), -1)
            return [preds, targets, losses]
        return preds

    def save(self, name='model'):
        model_path = os.path.join(self.model_path, f'{name}.pth')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict()}, model_path)
        print(f'Model is saved at {model_path}')
        
    def load(self, name='model', strict=True):
        model_path = os.path.join(self.model_path, f'{name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_state_dict = state_dict.get('model_state_dict', state_dict.get('state_dict'))
        optimizer_state_dict = state_dict.get('optimizer_state_dict')
        self.model.load_state_dict(model_state_dict)
        self.opt.load_state_dict(optimizer_state_dict)
        print(f'Model is loaded from {model_path}')

    def freeze_base(self):
        for layer in self.model.base.modules():
            if isinstance(layer , (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in layer.parameters(): p.requires_grad = True
            else:
                for p in layer.parameters(): p.requires_grad = False

    def freeze_head(self):
        for layer in self.model.head.modules():
            if isinstance(layer , (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in layer.parameters(): p.requires_grad = True
            else:
                for p in layer.parameters(): p.requires_grad = False

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def find_lr(self, mode='fastai'):
        self.opt = self.opt_func([{'params': self.model.base.parameters(), 'lr': 1e-6},
                                 {'params': self.model.head.parameters(), 'lr': 1e-6}])
        if mode == 'fastai':
            lr_finder = LRFinder(self.model, self.opt, self.loss_func, device='cuda')
            lr_finder.range_test(self.data.train_dl, end_lr=10, num_iter=100)
            lr_finder.plot()
            lr_finder.reset()
        if mode == 'leslie':
            lr_finder = LRFinder(self.model, self.opt, self.loss_func, device='cuda')
            lr_finder.range_test(self.data.train_dl, val_loader=self.data.val_dl, end_lr=1, num_iter=100, step_mode='linear')
            lr_finder.plot(log_lr=False)
            lr_finder.reset()
    
    def _set_lr(self, lr):
        lr_base, lr_head = lr
        self.opt.param_groups[0]['lr'] = lr_base
        self.opt.param_groups[1]['lr'] = lr_head
    
    def _define_discriminative_lr(self, lr):
        if isinstance(lr, float): return (lr/self.lr_div_factor, lr)
        if isinstance(lr, (tuple, list)): return lr[-2:]
        return (1e-4, 1e-3)

    def clip_grad(self, clip):
        old_clip = self.clip
        if isinstance(clip, (float, int)):
            self.clip = float(clip)
        print(f'Learner gradient clip value changes from {old_clip} -> {self.clip}')
        
    def fit_one_cycle(self, n_epochs, max_lr, name='model', **kwargs):
        """
        This function ignores the initial lr from the optimizer
        Using OneCycle Policy
        """
        max_lr = self._define_discriminative_lr(max_lr)
        lr_scheduler = partial(optim.lr_scheduler.OneCycleLR, max_lr=max_lr, epochs=n_epochs, 
                               steps_per_epoch=len(self.data.train_dl), **kwargs)
        self.fit(n_epochs=n_epochs, lr=None, lr_scheduler=lr_scheduler, name=name)

    def fit_sgd_warm(self, max_lr, cycle_len=3, cycle_mult=2, n_cycles=2, name='model', **kwargs):
        """
        This function ignores the initial lr from the optimizer
        Using Stochastic Gradient Descent with Warm Restarts
        """
        max_lr = self._define_discriminative_lr(max_lr)
        steps_per_cycle = len(self.data.train_dl)*cycle_len
        lr_scheduler = partial(optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=steps_per_cycle, 
                               T_mult=cycle_mult, **kwargs)
        n_epochs = cycle_len*(sum([cycle_mult**i for i in range(n_cycles)]))
        self.fit(n_epochs=n_epochs, lr=max_lr, lr_scheduler=lr_scheduler, name=name)
