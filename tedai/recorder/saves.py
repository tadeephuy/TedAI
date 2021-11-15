from ..imports import *
from ..utils import *
from . import TedRecorder

def config_model_save(recorder: TedRecorder, name='model', mode='reduce', monitor='val_loss', cycle_len=None):
    """
    save model function that with 4 modes: [improve, reduce, every_step, every_epoch]
    
    `improve`     : save model by checking the monitored value improvement after each validation
    `reduce`      : save model by checking the monitored value reduction after each validation
    `every_step`  : save model every `cycle_len` validation step
    `every_epoch` : save model every `cycle_len` epoch
    """
    if mode not in ['reduce', 'improve', 'every_step', 'every_epoch']:
        recorder.model_save = False
        print(f'No save setting mode found for {mode}, please choose `reduce`, `improve`, `every_epoch` or `every_step`')
        return
    
    recorder.model_save = True
    recorder.model_save_mode = mode
    recorder.model_save_name = name
    
    recorder.old_monitor_value = None

    if mode in ['reduce', 'improve']:
        recorder.model_save_monitor = monitor or 'val_loss'
        print(f'Save model by monitoring the {recorder.model_save_mode} of {recorder.model_save_monitor} after each validation step')
    elif mode == 'every_step':
        recorder.model_save_cycle_len = cycle_len or 1000
        print(f'Save model every {recorder.model_save_cycle_len} validation step')
    elif mode == 'every_epoch':
        recorder.model_save_cycle_len = cycle_len or 1
        print(f'Save model every {recorder.model_save_cycle_len} epoch')
TedRecorder.config_model_save = config_model_save

def model_save(recorder: TedRecorder):
    save_funcs = {
        'reduce':  recorder.__model_save_reduce,
        'improve': recorder.__model_save_improve,
        'every_step':  recorder.__model_save_step,
        'every_epoch':  recorder.__model_save_epoch,
    }
    save_funcs[recorder.model_save_mode]()
TedRecorder._model_save = model_save
    
def model_save_reduce(recorder: TedRecorder):
    monitor_value = recorder.epoch_report.at[len(recorder.epoch_report)-1, recorder.model_save_monitor]
    if monitor_value > (recorder.old_monitor_value or 10e6):
        return None
    recorder.old_monitor_value = monitor_value
    name = f'{recorder.model_save_name}-best-{recorder.model_save_monitor}'
    recorder.learner.save(name)
TedRecorder.__model_save_reduce = model_save_reduce
        
def model_save_improve(recorder: TedRecorder):
    monitor_value = recorder.epoch_report.at[len(recorder.epoch_report)-1, recorder.model_save_monitor]
    if monitor_value < (recorder.old_monitor_value or -10e6):
        return None
    recorder.old_monitor_value = monitor_value
    name = f'{recorder.model_save_name}-best-{recorder.model_save_monitor}'
    recorder.learner.save(name)
TedRecorder.__model_save_improve = model_save_improve

def model_save_step(recorder):
    pass
TedRecorder.__model_save_step = model_save_step

def model_save_epoch(recorder):
    pass
TedRecorder.__model_save_epoch = model_save_epoch
