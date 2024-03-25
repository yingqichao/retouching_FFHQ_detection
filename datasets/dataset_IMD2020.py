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


class IMD2020(AbstractDataset):
    """
    This dataset contains many folders.
    In each folder, there is a single original image named as "XXX_orig.jpg".
    And there are many tampered images derived from the original image named "XXX.png",
    and their corresponding mask image named "XXX_mask.png".
    """
    def __init__(self, opt, transform_list, tamp_list: str, mode: str):
        """
        :param crop_size: (H,W) or None
        :param tamp_list: '/root/MVSS-Net-master/data/IMD2020.txt'
        """
        super(IMD2020, self).__init__(crop_size=opt['datasets']['train']['GT_size'])
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
    root = dataset_root = Path("/data/IMD2020")

    imlist = []  
    # IMD2020
    for dir, _, files in os.walk(root):
        for file in files:
            filename = os.path.splitext(file)[0]
            if filename.endswith("mask"):
                tampered_image = filename.rsplit('_', 1)[0]
                if (Path(dir) / (tampered_image + ".png")).is_file():
                    imlist.append(','.join([str(Path(dir) / (tampered_image + ".png")), str(Path(dir) / file)]))
                elif (Path(dir) / (tampered_image + ".jpg")).is_file():
                    imlist.append(','.join([str(Path(dir) / (tampered_image + ".jpg")), str(Path(dir) / file)]))
                else:
                    raise KeyError("Uncorrect Picture!")

    print(len(imlist))  # 2010

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask = s.split(',')
        im_im = cv2.imread(str(im))
        mask_im = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        mask_im[mask_im > 0] = 1

        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1] or mask_im.sum() * 2 > mask_im.shape[0] * mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        im = im.split('/', 2)[-1]
        mask = mask.split('/', 2)[-1]
        new_imlist.append(','.join([im, mask]))

    with open("/root/DistillationForgertDetection/Distillation_model/Unet_with_MAE_Augmentation/datasets/IMD2020.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 2003

    
