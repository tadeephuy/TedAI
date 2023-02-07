from ..imports import *
from ..utils import *
from PIL import Image

__all__ = ['AlbuWrapper', 'OneInFiveCrop', 'one_in_five_crop', 
           'resize_aspect_ratio', 'custom_resize_func',
           'min_edge_crop']

def custom_resize_func(scale_ratio, interp, pil=True):
    """
    Create a function that accepts image size 
    to return a resize function,
    can be used for creating Augmentation Compose list.
    """
    def resize_func(img_size):
        x = [AlbuWrapper(albu.SmallestMaxSize, 
                max_size=img_size*scale_ratio, 
                interpolation=interp, 
                always_apply=True)]
        if pil: x = x + [ToPILImage()]
        return x
    return resize_func


class Identity:
    def __init__(self): pass
    def __call__(self, x): return x
    def __repr__(self): return 'Identity()'

class AlbuWrapper:
    def __init__(self, albu_class, **kwargs): self.albu_class = albu_class(**kwargs)
    def __call__(self, x):
        is_PIL = False
        if isinstance(x, Image.Image):
            is_PIL = True
            x = np.array(x)
        x = self.albu_class(image=x)['image']
        if is_PIL: x = Image.fromarray(x)
        return x
    def __repr__(self): return self.albu_class.__repr__()

class OneInFiveCrop:
    def __init__(self, size, pos):
        self.size, self.pos = size, pos

    def __repr__(self):
        return f"OneInFiveCrop(size={self.size}, pos={self.pos})"
    
    def __call__(self, x):
        return one_in_five_crop(x, self.size, self.pos)

def one_in_five_crop(x, size, pos='ct'):
    is_PIL = False
    if isinstance(x, Image.Image):
        is_PIL = True
        x = np.array(x)
    h,w,_ = x.shape

    ct_pos = [int((w - size)/2),        int((h - size)/2), 
              int((w - size)/2) + size, int((h - size)/2) + size]
    pos_dic = {
        'tl': [0, 0, size, size],
        'tr': [-size, 0, w, size],
        'ct': ct_pos,
        'bl': [0, -size, size, h],
        'br': [-size, -size, w, h]
    }
    pos = pos_dic.get(pos, ct_pos)
    x1, y1, x2, y2 = pos
    
    if is_PIL: return Image.fromarray(x[y1:y2,x1:x2])
    return x[y1:y2,x1:x2]

def resize_aspect_ratio(img, size, interp=cv2.INTER_AREA):
    """
    resize min edge to target size, keeping aspect ratio
    """
    if len(img.shape) == 2:
        h,w = img.shape
    elif len(img.shape) == 3:
        h,w,_ = img.shape
    else:
        return None
    if h > w:
        new_w = size
        new_h = h*new_w//w
    else:
        new_h = size
        new_w = w*new_h//h
    return cv2.resize(img, (new_w, new_h), interpolation=interp)     

def min_edge_crop(img, position="center"):
    """
    crop image base on min size
    :param img: image to be cropped
    :param position: where to crop the image
    :return: cropped image
    """
    assert position in ['center', 'left', 'right'], "position must either be: left, center or right"

    h, w = img.shape[:2]

    if h == w:
        return img

    min_edge = min(h, w)
    if h > min_edge:
        if position == "left":
            img = img[:min_edge]
        elif position == "center":
            d = (h - min_edge) // 2
            img = img[d:-d] if d != 0 else img

            if h % 2 != 0:
                img = img[1:]
        else:
            img = img[-min_edge:]

    if w > min_edge:
        if position == "left":
            img = img[:, :min_edge]
        elif position == "center":
            d = (w - min_edge) // 2
            img = img[:, d:-d] if d != 0 else img

            if w % 2 != 0:
                img = img[:, 1:]
        else:
            img = img[:, -min_edge:]

    assert img.shape[0] == img.shape[1], f"height and width must be the same, currently {img.shape[:2]}"
    return img

from .tta import *