from ..imports import *
from ..utils import *
from . import TedRecorder

def plot_losses(recorder: TedRecorder):
    """ Plot train loss and valid loss
    """
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111)
    ax.yaxis.set_ticks(np.arange(-0.5, 2, 0.1))
    x_train_loss = [c[0] for c in recorder.train_loss]
    y_train_loss = [c[1] for c in recorder.train_loss]
    plt.plot(x_train_loss, y_train_loss, label='train')
    x_val_loss = [c[0] for c in recorder.val_loss]
    y_val_loss = [c[1] for c in recorder.val_loss]
    plt.plot(x_val_loss,y_val_loss, label='valid')
    plt.xlabel("steps")
    plt.ylabel("Loss")
    ax.legend()
    ax.grid(True, 'both', linestyle='dotted')
    plt.title('Train Loss & Validation Loss', {'fontsize': 20})
TedRecorder.plot_losses = plot_losses

def plot_metrics(recorder: TedRecorder):
    """ Plot metrics value
    """
    fig = plt.figure(figsize=(10,7)); ax = fig.add_subplot(111)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    plt.xlabel("steps")
    plt.ylabel("value")
    for mtrc, values in recorder.metrics.items():
        x_mtrc = [c[0] for c in values]
        y_mtrc = [c[1] for c in values]
        plt.plot(x_mtrc, y_mtrc, label=mtrc)
    ax.legend()
    ax.grid(True, 'major', linestyle='dotted')
    plt.title('Metrics', {'fontsize': 20})
TedRecorder.plot_metrics = plot_metrics

def plot_lr(recorder: TedRecorder, layer='head'):
    """ Plot learning rate during training, layer can be either [`head`, 'base', `all`]
    """
    plt.figure(figsize=(10,7))
    plt.xlabel("steps")
    plt.ylabel("lr")
    x_lr = [c[0] for c in recorder.lr]
    if layer=='head' or layer=='all':
        y_lr = [c[1][-1] for c in recorder.lr]
        plt.plot(x_lr, y_lr, label='head')
        plt.legend()
    if layer=='base' or layer=='all': 
        y_lr = [c[1][0] for c in recorder.lr]
        plt.plot(x_lr, y_lr, label='base')
        plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0.01, 10)) 
    plt.grid(True, linestyle='dotted')
    plt.title('Learning Rate', {'fontsize': 20})
TedRecorder.plot_lr = plot_lr