from ..imports import *
from ..utils import *
from .. import TedRecorder

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
                loss = self.loss_func(out, yb)#.to(self.device)
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
                loss = self.loss_func(self.model(xb), yb)#.to(self.device)
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
 
            # save model and optimizer states
            self.save(name=name, log=False)

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

    def save(self, name='model', log=True):
        os.makedirs(self.model_path, exist_ok=True)
        model_path = os.path.join(self.model_path, f'{name}.pth')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict()}, model_path)
        if log:
            print(f'Model is saved at {model_path}')
        
    def load(self, name='model'):
        model_path = os.path.join(self.model_path, f'{name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_state_dict = state_dict.get('model_state_dict', state_dict.get('state_dict'))
        optimizer_state_dict = state_dict.get('optimizer_state_dict')
        self.model.load_state_dict(model_state_dict)
        self.opt.load_state_dict(optimizer_state_dict)
        print(f'Model is loaded from {model_path}')
    
    def set_loss_func(self, loss_func):
        self.loss_func = loss_func.to(self.device)
    
    def set_opt_func(self, opt_func):
        self.opt_func = opt_func
        self.opt = self.opt_func([{'params': self.model.base.parameters(), 'lr': 1e-4},
                                 {'params': self.model.head.parameters(), 'lr': 1e-3}])
        
    
    def _set_lr(self, lr):
        lr_base, lr_head = lr
        self.opt.param_groups[0]['lr'] = lr_base
        self.opt.param_groups[1]['lr'] = lr_head
    
    def _define_discriminative_lr(self, lr):
        if isinstance(lr, float): return (lr/self.lr_div_factor, lr)
        if isinstance(lr, (tuple, list)): return lr[-2:]
        return (1e-4, 1e-3)

from .utils import *
from .fits import *