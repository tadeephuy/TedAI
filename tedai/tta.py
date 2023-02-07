from .imports import *
from .utils import *
from .data.tta import TestTimeAugmentationData, TTAData
from .augs.tta import *

def optimize_tta_weights(preds_tta, gt, criterion, n_classes=1, 
                         thresh_percentile=0.9,
                         device='cuda',
                         init_gain=1.0,
                         n_iters=20_000, lr=1e-1, 
                         momentum=0.9, weight_decay=1e-4,
                         patience=500):
    """Optimize the tta weights from `preds_tta`.
    Arguments:
        - preds_tta: the predictions of each tta combination, in order.
        - gt: groundtruth
        - criterion: loss function for optimzation, recommened to use pytorch's `BCEWithLogitsLoss` for binary classification or `CrossEntropyLoss` for multiclass classification.
    
    Return: 
        - tta_weights: raw optimized TTA weights 
        - tta_configs: the recommened tta config dictionary, key: the index of the tta combination, v: its weight.
    """
    
    n = len(gt) # num_samples
    t = int(len(preds_tta)//n) # num_tta_combinations
    
    # initialization
    tta_weights = torch.empty([t,1,n_classes], device=device)
    nn.init.xavier_uniform_(tta_weights, gain=init_gain)
    tta_weights = nn.Parameter(data=tta_weights, requires_grad=True)

    preds_tta_tensor = torch.tensor(preds_tta, requires_grad=False, device=device)
    preds_tta_tensor = preds_tta_tensor.view(t, n, n_classes)

    y = torch.tensor(gt, dtype=float, requires_grad=False, device=device).unsqueeze(-1)
    
    # optimization
    optimizer = torch.optim.SGD(params=[tta_weights], lr=lr, 
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    criterion = criterion.to(device)

    bar = progress_bar(range(n_iters))
    for _ in bar:
        optimizer.zero_grad()

        x = preds_tta_tensor*tta_weights
        x = x.sum(axis=0)
        loss = criterion(x, y)
        loss.backward()

        optimizer.step()
        scheduler.step(loss.item())
        with torch.no_grad():
            tta_weights.clamp_(min=0)

        bar.comment = f'{loss.item():.2E}'
    
    # calculate threshold to zero out insignificant ttas
    tta_weights = tta_weights.detach().cpu().numpy()
    
    if thresh_percentile == 1:
        tta_thresh = 0
    else:
        sorted_tta_w = np.sort(tta_weights, axis=0)[::-1]
        tta_thresh_idx = np.where((np.cumsum(sorted_tta_w)/tta_weights.sum())  > thresh_percentile)[0][0]
        tta_thresh = sorted_tta_w[tta_thresh_idx].item() - 1e-5

    tta_w = tta_weights[:,0,0]
    
    # add the weight of those insignificant ttas to the remainings
    rest = tta_w[tta_w<tta_thresh].sum()
    tta_idxes = np.where(tta_w>=tta_thresh)[0]
    rest_w = tta_w[tta_idxes]/tta_w[tta_idxes].sum()
    
    tta_configs = {idx: tta_w[idx] + rest*rest_w[i] 
                   for i, idx in enumerate(tta_idxes)}
    
    # visualization
    plt.bar(range(t), tta_w)
    plt.axhline(y=tta_thresh, color='red')
    plt.xticks(range(t))
    plt.show()

    return tta_weights, tta_configs