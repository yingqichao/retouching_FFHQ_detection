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
import re


class COVER(AbstractDataset):
    """
    directory structure:
    COVER
    │── image => "(number).tif" means the original image, "(number)t.tif" means the forged image.
    └── mask => "(number)forged.tif" means the corresponding mask
    """
    def __init__(self, opt, transform_list, tamp_list: str, mode: str):
        """
        :param crop_size: (H,W) or None
        :param tamp_list: '/root/MVSS-Net-master/data/COVER.txt'
        """
        super(COVER, self).__init__(crop_size=opt['datasets']['train']['GT_size'])
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
    # NIST16
    tamp_root = root / "Cover/image"
    mask_root = root / "Cover/mask"
    for file in os.listdir(tamp_root):
        filename = os.path.splitext(file)[0]
        if filename.endswith('t'):
            number = re.findall(r"\d+", filename)[0]
            imlist.append(','.join([str(Path("Cover/image") / file),
                                    str(Path("Cover/mask") / (number + "forged.tif"))]))
        assert (mask_root / (number + "forged.tif")).is_file()

    print(len(imlist))  # 100

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask = s.split(',')
        im_im = cv2.imread(str(root / im)) / 255.0
        mask_im = cv2.imread(str(root / mask), cv2.IMREAD_GRAYSCALE) / 255.0 
        mask_im[mask_im > 0] = 1
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1] or mask_im.sum() * 2 > mask_im.shape[0] * mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        new_imlist.append(s)

    with open("/root/DistillationForgertDetection/Distillation_model/Unet_with_MAE_Augmentation/datasets/COVER.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 91

    
