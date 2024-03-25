import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DualRandomHorizonalFlip:
    def __init__(self, p=0.5):
        self.p=p
    
    def __call__(self, img, img_store, mask):
        if torch.rand(1) < self.p:
            img = TF.hflip(img)
            img_store = TF.hflip(img_store)
            mask = TF.hflip(mask)
        return img, img_store, mask

class DualRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p=p

    def __call__(self, img, img_store, mask):
        if torch.rand(1) < self.p:
            img = TF.vflip(img)
            img_store = TF.vflip(img_store)
            mask = TF.vflip(mask)
        return img, img_store, mask

class DualResize:
    # pil_modes_mapping = {
    #     InterpolationMode.NEAREST: 0,
    #     InterpolationMode.BILINEAR: 2,
    #     InterpolationMode.BICUBIC: 3,
    #     InterpolationMode.BOX: 4,
    #     InterpolationMode.HAMMING: 5,
    #     InterpolationMode.LANCZOS: 1,
    # }
    def __init__(self, crop_size, interpolation=3):
        self.crop_size = crop_size
        self.interpolation = interpolation
    
    def __call__(self, img, img_store, mask):
        if self.crop_size is not None:
            img = TF.resize(img, self.crop_size)
            img_store = TF.resize(img_store, self.crop_size)
            mask = TF.resize(mask, self.crop_size)
        return img, img_store, mask

class DualTo_tensor:
    def __call__(self, img, img_store, mask):
        img = TF.to_tensor(img).float()
        img_store = TF.to_tensor(img_store).float()
        mask = TF.to_tensor(mask).int()
        return img, img_store, mask