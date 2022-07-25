from ..imports import *
from ..utils import *
from . import TedLearner
from .inference import TedInference

def get_preds(learn: TedLearner, mode='test', with_losses=False, activ=None):
    """
    get all predictions of a given dataset by specifying mode `train`, `valid` or `test`
    with_losses:
        True: the dataset has labels, get the losses and label -> (preds, labels, losses)
        False: only get predictions -> preds
    """
    learn.model.eval()
    preds, targets, losses = [], [], []
    ds = {
        'train': learn.data.train_ds,
        'valid': learn.data.val_ds,
        'test': learn.data.test_ds or learn.data.val_ds
    }.get(mode, learn.data.val_ds)
    dl = learn.data._create_dl(dataset=ds, shuffle=False, bs=None if not with_losses else 1)
    with torch.no_grad():
        for batch in progress_bar(dl):
            if mode=='test': # no label
                xb = batch.to(learn.device)
            else:
                xb = batch[0].to(learn.device)
            out = learn.model(xb)
            if activ is not None: out = activ(out)
            preds.append(out.cpu().numpy())
            if len(batch) == 2 and with_losses:
                yb = batch[1].to(learn.device)
                loss = learn.loss_func(out, yb).cpu().numpy()
                target = yb.cpu().numpy()
                targets.append(target)
                losses.append(loss)
    preds = np.concatenate(preds, axis=0)
    if len(batch) == 2 and with_losses:
        targets = np.concatenate(targets, axis=0)
        losses = np.expand_dims(np.array(losses), -1)
        return [preds, targets, losses]
    return preds
TedLearner.get_preds = get_preds
TedInference.get_preds = get_preds

NORMS = [
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LayerNorm
]

def freeze_base(learn: TedLearner, freeze_bn=False, norms=NORMS):
    learn.unfreeze()
    print(f'freeze base {"but not Norms" if not freeze_bn else ""}')
    is_norm = lambda x: any([isinstance(x, c) for c in norms])
    for layer in learn.model.base.modules():
        if is_norm(layer) and not freeze_bn:
            for p in layer.parameters(): p.requires_grad = True
        else:
            for p in layer.parameters(): p.requires_grad = False
TedLearner.freeze_base = freeze_base

def freeze_head(learn: TedLearner, norms=NORMS):
    learn.unfreeze()
    is_norm = lambda x: any([isinstance(x, c) for c in norms])
    for layer in learn.model.head.modules():
        if is_norm(layer):
            for p in layer.parameters(): p.requires_grad = True
        else:
            for p in layer.parameters(): p.requires_grad = False
TedLearner.freeze_head = freeze_head

def unfreeze(learn: TedLearner):
    for p in learn.model.parameters():
        p.requires_grad = True
TedLearner.unfreeze = unfreeze

def find_lr(learn: TedLearner, mode='fastai'):
    learn.opt = learn.opt_func([{'params': learn.model.base.parameters(), 'lr': 1e-6},
                                {'params': learn.model.head.parameters(), 'lr': 1e-6}])
    if mode == 'fastai':
        lr_finder = LRFinder(learn.model, learn.opt, learn.loss_func, device='cuda')
        lr_finder.range_test(learn.data.train_dl, end_lr=10, num_iter=100)
        lr_finder.plot()
        lr_finder.reset()
    if mode == 'leslie':
        lr_finder = LRFinder(learn.model, learn.opt, learn.loss_func, device='cuda')
        lr_finder.range_test(learn.data.train_dl, val_loader=learn.data.val_dl, end_lr=1, num_iter=100, step_mode='linear')
        lr_finder.plot(log_lr=False)
        lr_finder.reset()
TedLearner.find_lr = find_lr

def clip_grad(learn: TedLearner, clip):
    old_clip = learn.clip
    if isinstance(clip, (float, int)):
        learn.clip = float(clip)
    print(f'Learner gradient clip value changes from {old_clip} -> {learn.clip}')
TedLearner.clip_grad = clip_grad
