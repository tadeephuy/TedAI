import os
import collections
import gc
import torch
from functools import partial
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, HTML, clear_output
from fastprogress.fastprogress import master_bar, progress_bar