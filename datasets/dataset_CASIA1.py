import os
import numpy as np
from PIL import Image

from .AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image, ImageChops, ImageFilter
import torch
from pathlib import Path
import cv2


class CASIA1(AbstractDataset):
    """
    directory structure:
    CASIA1
    │── Au
    │── Tp
      │—— CM
      └── SP
    └── Groundtruth : Add suffix "_gt.png"
      │—— CM
      └── SP
    """
    def __init__(self, opt, transform_list, tamp_list: str, mode: str):
        super(CASIA1, self).__init__(crop_size=opt['datasets']['train']['GT_size'])
        self.opt = opt
        """
        :param transform_list: Image Enhancement 
        :param tamp_list: path of txt file
        """
        self.transform_list = transform_list
        self._root_path = Path("/hotdata")
        self.mode = mode

        with open(tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0] 
        mask_path = self._root_path / self.tamp_list[index][1]
        
        return self._create_tensor(tamp_path, mask_path, self.transform_list, self.mode)


if __name__ == '__main__':
    root = dataset_root = Path("/data")

    imlist = []  
    # CASIA1
    tamp_root = root / "CASIA1/Tp"
    mask_root = root / "CASIA1/Groundtruth"
    for dir, _, files in os.walk(tamp_root):
        for file in files:
            if file in ['Sp_D_NRN_A_cha0011_sec0011_0542.jpg']:
                continue
            imlist.append(','.join([str(Path(dir) / file),
                            str(Path(mask_root / str(dir).rsplit('/', 1)[-1]) / (os.path.splitext(file)[0] + "_gt.png"))]))
            assert(Path(mask_root / str(dir).rsplit('/', 1)[-1]) / (os.path.splitext(file)[0] + "_gt.png")).is_file()
    
    print(len(imlist))  # 920

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask = s.split(',')
        im_im = cv2.imread(str(im)) / 255.0
        mask_im = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE) / 255.0
        mask_im[mask_im > 0] = 1
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1] or mask_im.sum() * 2 > mask_im.shape[0] * mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        im = im.split('/', 2)[-1]
        mask = mask.split('/', 2)[-1]
        new_imlist.append(','.join([im, mask]))

    with open("/root/DistillationForgertDetection/Distillation_model/Unet_with_MAE_Augmentation/datasets/CASIA1.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 917
