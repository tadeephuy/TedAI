from ..imports import *
from ..utils import *
from . import TedLearner

__all__ = ['fit_one_cycle', 'fit_sgd_warm']

def fit_one_cycle(learn: TedLearner, n_epochs, max_lr, name='model', **kwargs):
    """
    This function ignores the initial lr from the optimizer
    Using OneCycle Policy
    """
    max_lr = learn._define_discriminative_lr(max_lr)
    lr_scheduler = partial(optim.lr_scheduler.OneCycleLR, max_lr=max_lr, epochs=n_epochs, 
                            steps_per_epoch=len(learn.data.train_dl), **kwargs)
    learn.fit(n_epochs=n_epochs, lr=None, lr_scheduler=lr_scheduler, name=name)
TedLearner.fit_one_cycle = fit_one_cycle

def fit_sgd_warm(learn: TedLearner, max_lr, cycle_len=3, cycle_mult=2, n_cycles=2, name='model', **kwargs):
    """
    This function ignores the initial lr from the optimizer
    Using Stochastic Gradient Descent with Warm Restarts
    """
    max_lr = learn._define_discriminative_lr(max_lr)
    steps_per_cycle = len(learn.data.train_dl)*cycle_len
    lr_scheduler = partial(optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=steps_per_cycle, 
                            T_mult=cycle_mult, **kwargs)
    n_epochs = cycle_len*(sum([cycle_mult**i for i in range(n_cycles)]))
    learn.fit(n_epochs=n_epochs, lr=max_lr, lr_scheduler=lr_scheduler, name=name)
TedLearner.fit_sgd_warm = fit_sgd_warm
