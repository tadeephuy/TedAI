from ..imports import *
from . import Identity, OneInFiveCrop, AlbuWrapper
from torchvision import transforms as torch_trns

__all__ = ['cxr_ttas', 'cxr_ttas_v2']

cxr_ttas = [
    [
        Identity(),
        torch_trns.RandomHorizontalFlip(p=1.0),
    ],
    [
        Identity(),
        AlbuWrapper(albu.CLAHE, clip_limit=(1.5, 1.5), always_apply=True, p=1.0),
    ],
    [
        OneInFiveCrop(size=512, pos='ct'),
        OneInFiveCrop(size=512, pos='tl'),
        OneInFiveCrop(size=512, pos='tr'),
        OneInFiveCrop(size=512, pos='bl'),
        OneInFiveCrop(size=512, pos='br'),
    ],
    [ToTensor()],
    [Normalize(mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225])],
]

cxr_ttas_v2 = [
    [
        Identity(),
        torch_trns.RandomHorizontalFlip(p=1.0),
    ],
    [
        Identity(),
        AlbuWrapper(albu.Sharpen, alpha=(0.2, 0.2), lightness=(0.6, 0.6), always_apply=True, p=1.0),
        AlbuWrapper(albu.Equalize, mode='cv', always_apply=True, p=1.0),
    ],
    [
        OneInFiveCrop(size=512, pos='ct'),
        OneInFiveCrop(size=512, pos='tl'),
        OneInFiveCrop(size=512, pos='tr'),
        OneInFiveCrop(size=512, pos='bl'),
        OneInFiveCrop(size=512, pos='br'),
    ],
    [ToTensor()],
    [Normalize(mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225])],
]