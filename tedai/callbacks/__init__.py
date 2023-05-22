from ..imports import *
from ..utils import *
from ..learner import TedLearner

class CallbacksHandler:
    def __init__(self, learn, cbs=[]):
        self.learn = learn
        self.cbs = cbs
    
    def on_train_begin(self):
        [cb.on_train_begin() for cb in self.cbs]
    
    def on_epoch_begin(self):
        self.learn.model.train()
        self.learn.train_bar = progress_bar(self.learn.data.train_dl, parent=self.learn.master_bar)
        [cb.on_epoch_begin() for cb in self.cbs]
    
    def on_batch_begin(self, xb, yb):
        self.xb, self.yb = xb, yb
        for cb in self.cbs:
            self.xb, self.yb = cb.on_batch_begin(self.xb, self.yb)
        return self.xb, self.yb
    
    def on_preds_begin(self, xb):
        self.xb = xb
        for cb in self.cbs:
            self.xb = cb.on_preds_begin(self.xb)
        return self.xb
    
    def on_loss_begin(self, pb, yb):
        self.pb, self.yb = pb, yb
        for cb in self.cbs:
            self.pb, self.yb = cb.on_loss_begin(self.pb, self.yb)
        return self.pb, self.yb
    
    def on_backward_begin(self, loss):
        self.loss = loss
        for cb in self.cbs:
            self.loss = cb.on_backward_begin(self.loss)
        return self.loss
    
    def on_step_begin(self):
        [cb.on_step_begin() for cb in self.cbs]
    
    def on_batch_end(self):
        [cb.on_batch_end() for cb in self.cbs]
    
    def on_validation_begin(self):
        [cb.on_validation_begin() for cb in self.cbs]
        
    def on_validation_end(self, val_loss, metrics):
        self.val_loss, self.metrics = val_loss, metrics
        for cb in self.cbs:
            self.val_loss, self.metrics = cb.on_validation_end(self.val_loss, self.metrics)
        return self.val_loss, self.metrics
    
    def on_epoch_end(self):
        [cb.on_epoch_end() for cb in self.cbs]

    def on_train_end(self):
        [cb.on_epoch_end() for cb in self.cbs]
    
class Callback:
    """Placeholder for Callbacks implementation"""
    def __init__(self, learn): self.learn = learn

    def on_train_begin(self): pass
    
    def on_epoch_begin(self): pass
    
    def on_batch_begin(self, xb, yb): return xb, yb 
    
    def on_preds_begin(self, xb): return xb
    
    def on_loss_begin(self, pb, yb): return pb, yb
    
    def on_backward_begin(self, loss): return loss
    
    def on_step_begin(self): pass
    
    def on_batch_end(self): pass
    
    def on_validation_begin(self): pass
        
    def on_validation_end(self, val_loss, metrics): return val_loss, metrics
    
    def on_epoch_end(self): pass

    def on_train_end(self): pass

def set_callbacks(self, callbacks):
    self.callbacks_handler = CallbacksHandler(self, callbacks)
TedLearner.set_callbacks = set_callbacks

def fit_with_callbacks(self, n_epochs, lr=None, lr_scheduler=None, name='model', **kwargs):
    self._reset_state(n_epochs)
    if lr is not None: self._set_lr(lr)
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(self.opt, **kwargs)
    else:
        last_lr = None
    
    cbs = self.callbacks_handler

    # train
    cbs.on_train_begin()
    train_loss_value = AverageMeter()
    for epoch in self.master_bar:
        cbs.on_epoch_begin()
        self.model.train()
        self.train_bar = progress_bar(self.data.train_dl, parent=self.master_bar)
        for j, (xb, yb) in enumerate(self.train_bar):
            xb, yb = cbs.on_batch_begin(xb, yb)
            xb, yb = xb.to(self.device), yb.to(self.device)
            
            xb = cbs.on_preds_begin(xb)
            pb = self.model(xb)
            
            pb, yb = cbs.on_loss_begin(pb, yb)
            loss = self.loss_func(pb, yb)
            
            loss = cbs.on_backward_begin(loss)
            loss.backward()
            
            cbs.on_step_begin()
            self.opt.step()
            self.opt.zero_grad()

            if lr_scheduler is not None:
                last_lr = lr_scheduler.get_last_lr()
                lr_scheduler.step()
            
            
            train_loss_value.update(val=loss.item())
            train_loss = train_loss_value.avg
            self.recorder.update(train_loss=train_loss, lr=last_lr, validate_every=self.validate_every)

            cbs.on_batch_end()

        # save model and optimizer states
        self.save(name=name, log=False)

        # validation
        cbs.on_validation_begin()
        val_loss, metrics = self.validate()
        
        val_loss, metrics = cbs.on_validation_end(val_loss, metrics)
        self.recorder._update(val_loss=val_loss, metrics=metrics, done_epoch=True)
        self.recorder.report_epoch(done_epoch=True)
        
        cbs.on_epoch_end()

    ## on train end
    cbs.on_epoch_end()
    # save model and optimizer states
    self.save(name=name)