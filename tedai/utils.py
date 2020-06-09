import torch
from torch import nn
import torchvision
from torchvision.transforms import *
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial
import numpy as np
from . import *
from .metrics import *

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Denormalize(nn.Module):
    """
    !! not a transform function !!
    Batch denormalization Module
    Can be use for image showing
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), device=torch.device('cpu')):
        super(Denormalize, self).__init__()
        self.mean,self.std=torch.FloatTensor(mean).to(device),torch.FloatTensor(std).to(device)
    def forward(self, x):
        return x*self.mean.view(1, 3, 1, 1) + self.std.view(1, 3, 1, 1)

def make_imgs(xb, n_row=8, denorm=Denormalize(), device=torch.device('cpu'), plot=True):
    xb = xb[:n_row].to(device)
    xb = denorm(xb) 
    grid_img = torchvision.utils.make_grid(xb.cpu(), nrow=n_row, normalize=True, pad_value=1)
    grid_img = grid_img.permute(1, 2, 0)
    if not plot: 
        return ToPILImage()((grid_img.numpy()*255).astype(np.uint8))
    plt.close(); plt.figure(figsize=(30,30)); plt.imshow(grid_img); plt.show()

def create_dataset(dataset_class, data_path, df, **kwargs):
    return partial(dataset_class, data_path=data_path, df=df, **kwargs)

def create_transforms(zoom_in_scale=1.3, max_rotate=12, vert_flip=False, normalize=True, p_flip=0.4, p_affine=0.45, p_color=0.2):
    """
    create basic transforms for training data
    """
    def Identity(x): return x

    return lambda img_size: Compose([
        ToPILImage(),
        Resize(int(img_size*zoom_in_scale)),
        RandomResizedCrop(img_size, (0.9, 1.15)),
        RandomHorizontalFlip(p=p_flip),
        RandomVerticalFlip(p=p_flip) if vert_flip else Lambda(Identity),
        RandomApply([RandomAffine(max_rotate, (0.05, 0.05), (0.95, 1.25), 
                    (0.2, 0.2, 0.2, 0.2), 2)], p=p_affine),
        RandomApply([ColorJitter(brightness=(0.2), contrast=(0.85, 1.0))], p=p_color),
        RandomPerspective(0.05, p=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize else Lambda(Identity)
    ])

def report_distribution(df, label_cols_list, sort=True):
    distribution_report = {}
    for c in label_cols_list:
        distribution_report[c] = df[c].value_counts()
    distribution_report = pd.DataFrame.from_dict(distribution_report, orient='index').fillna(0)
    distribution_report.columns = [f'{c:0.2f}' for c in list(distribution_report.columns)]
    if sort:
        distribution_report = distribution_report.reindex(sorted(distribution_report.columns), axis=1)
    return distribution_report.applymap(int)

def report_binary_thresholded_metrics(y_pred, y_true, thresh_step=0.1):
    report = pd.DataFrame(columns=['precision', 'recall', 'specificity', 
                                   'accuracy', 'auc', 'f2', 'f1', 'TP', 'FP', 'TN', 'FN'])
    preds = y_pred
    for thresh in np.arange(0.00, 1.00, thresh_step):
        y_pred = (preds >= thresh).astype(np.uint8).squeeze()
        metrics = binary_metrics(y_pred, y_true, log=False)
        counts = classification_counts(y_pred, y_true, log=False)
        row = pd.DataFrame.from_dict({f'{thresh:0.2f}': dict(metrics, **counts)}, orient='index')
        report = pd.concat([report, row], ignore_index=False)
        report.index.name = 'threshold'
    return report

def report_wrong_samples(y_pred, y_true, df, loss=None, top_k=None, full=False, thresh=None):
    """
    `loss` must be not None to use top_k
    """
    assert len(y_true) == len(y_pred)
    y_pred_org = y_pred.copy()
    y_true = y_true.astype(np.uint8)
    losses = np.full(len(y_true), fill_value=None) if loss is None else loss
    if thresh is not None: 
        y_pred = (y_pred >= thresh).astype(np.uint8)
        wrong_idxes = [idx for idx, pred in enumerate(y_pred) if y_true[idx] != pred]
    else:
        wrong_idxes = range(len(y_pred))
    report = df.copy(deep=True)
    report = report[['Images']]
    report['preds'] = y_pred_org
    report['label'] = y_true
    if not full:
        report = report.filter(wrong_idxes, axis='index')
        losses = losses[wrong_idxes]
    if loss is not None:
        report['losses'] = losses
    if top_k is None:
        return report
    assert loss is not None, 'No loss to sort top K'
    report = report.sort_values(by='losses', ascending=False)
    return report[:top_k]

def read_csv(path, normalize_columns=True, thresh=0.5, **kwargs):
    df = pd.read_csv(path, **kwargs)
    if normalize_columns:
        df.columns = [c.replace(' ', '_') for c in df.columns]
    if thresh is not None:
        df[df.columns[1:]] = (df[df.columns[1:]].values >= thresh).astype(np.uint8) 
    return df

def count_params(model, trainable=False):
    if trainable: return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
