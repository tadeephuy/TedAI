from ..imports import *
from ..utils import *
from . import TedRecorder

def show_imgs(recorder: TedRecorder, imgs):
    if not hasattr(recorder.master_bar, 'imgs_out'):
        recorder.master_bar.imgs_out = display(imgs, display_id=True)
    else: 
        recorder.master_bar.imgs_out.update(imgs)
TedRecorder.show_imgs = show_imgs

def show_report(recorder: TedRecorder, mode='all'):
    """ show training report, mode can be either [`all`, `epoch`, `step`]
    """
    mode = 'all' if mode not in ['all', 'epoch', 'step'] else mode
    
    df = recorder.epoch_report.copy(deep=True)
    if mode == 'all':
        display(HTML(df.to_html(index=False))); return
    
    if mode in ['epoch', 'step']:
        df['mode'] = df['step/epoch'].apply(lambda x: 'epoch' if 'epoch' in x else 'step')
        df = df[df['mode'] == mode].drop('mode', axis=1)
        display(HTML(df.to_html(index=False))); return
TedRecorder.show_report = show_report