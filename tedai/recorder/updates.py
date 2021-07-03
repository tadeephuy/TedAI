from ..imports import *
from ..utils import *
from . import TedRecorder

def _update(recorder, train_loss=None, val_loss=None, metrics=None, lr=None, done_epoch=False):
    recorder._update_train_loss(train_loss)
    recorder._update_lr(lr)
    if done_epoch:
        recorder.step -= 1

    recorder._update_val_loss(val_loss)
    recorder._update_metrics(metrics)

    if done_epoch:
        recorder._update_epoch_report(recorder.train_loss[-1][-1], val_loss, metrics, mode='epoch')
        recorder.epoch += 1
    else:
        recorder._update_epoch_report(train_loss, val_loss, metrics, mode='step')
    recorder.step += 1
TedRecorder._update = _update

def _update_train_loss(recorder, train_loss):
    if train_loss is not None:
        recorder.train_loss.append((recorder.step, train_loss))
TedRecorder._update_train_loss = _update_train_loss

def _update_val_loss(recorder, val_loss):
    if val_loss is not None:
        recorder.val_loss.append((recorder.step, val_loss))
TedRecorder._update_val_loss = _update_val_loss

def _update_metrics(recorder, metrics):
    if metrics is not None:
        for i, metric_name in enumerate(recorder.metrics.keys()):
            recorder.metrics[metric_name].append((recorder.step, metrics[metric_name]))
TedRecorder._update_metrics = _update_metrics

def _update_lr(recorder, lr):
    if lr is not None:
        recorder.lr.append((recorder.step, lr))
TedRecorder._update_lr = _update_lr

def _update_epoch_report(recorder, train_loss, val_loss, metrics, mode='epoch'):
    if mode == 'step' and val_loss is None: return
    index = {'epoch': f'epoch_{recorder.epoch}', 'step': f'{recorder.step}'}[mode]
    
    row = pd.DataFrame([
        [index, train_loss, val_loss, 
        *[metric_value for _, metric_value in (metrics or {}).items()]]], 
        columns=recorder.epoch_report.columns)
    recorder.epoch_report = recorder.epoch_report.append(row, ignore_index=True)
    recorder.str_stats = [index, f'{train_loss:.6f}', f'{val_loss:.6f}', *[f'{metric_value:.6f}' for _, metric_value in (metrics or {}).items()]]
TedRecorder._update_epoch_report = _update_epoch_report