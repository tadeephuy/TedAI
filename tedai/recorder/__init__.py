from ..imports import *
from ..utils import *

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
    
    def update(self, train_loss, lr, validate_every=None):
        self.learner.train_bar.comment = f'train_loss: {train_loss:0.6f}'
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

from .plots import *
from .utils import *
from .saves import *
from .updates import *