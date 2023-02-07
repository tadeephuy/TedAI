import os
import collections
import gc
import torch
from functools import partial
from torch import nn, optim
from torch.nn import functional as F
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder
import cv2
import albumentations as albu
import pandas as pd
import numpy as np
from collections import namedtuple
import pickle as pkl
from matplotlib import pyplot as plt
from IPython.display import display, HTML, clear_output
from fastprogress.fastprogress import master_bar, progress_bar

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]