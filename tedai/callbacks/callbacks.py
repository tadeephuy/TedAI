import torch
from torch import nn
import numpy as np
from . import *

class GradientClippingCallback(Callback):
    def __init__(self, learn, clip=0.1):
        self.learn,  self.clip = learn, clip
    def on_step_begin(self):
        nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)
        
class ShowTrainingImagesCallback(Callback):
    __oder = 999
    def __init__(self, learn, n_row=8, denorm=Denormalize(), device='cpu'):
        self.learn, self.n_row, self.denorm, self.device = learn, n_row, denorm, device
    
    def on_batch_begin(self, xb, yb):
        self.xb_to_show = xb.detach()
        return xb, yb

    def on_epoch_end(self):
        img_to_show = make_imgs(self.xb_to_show, n_row=self.n_row, 
                                denorm=self.denorm, device=self.device, plot=False)
        self.learn.recorder.show_imgs(img_to_show)
        
class MixUpCallback(Callback):
    def __init__(self, learn, alpha=0.1):
        self.learn, self.alpha = learn, alpha

    @staticmethod
    @torch.no_grad()
    def mixup_xb_and_yb(xb, yb, alpha):
        lam = np.random.beta(alpha, alpha)
        indices = np.random.permutation(xb.size(0))
        xb = xb*lam + xb[indices]*(1-lam)
        yb_ = yb[indices]
        return xb, yb_, lam

    def on_batch_begin(self, xb, yb):
        self.yb = yb
        xb, yb_, self.lam = self.mixup_xb_and_yb(xb, yb, self.alpha)
        
        self.extra_loss = None
        return xb, yb_

    def on_loss_begin(self, pb, yb):
        self.extra_loss = self.learn.loss_func(pb, self.yb.to(pb.device)) # original yb
        return pb, yb # mixed yb
    
    def on_backward_begin(self, loss):
        loss = loss*(1-self.lam) + self.extra_loss*self.lam
        return loss