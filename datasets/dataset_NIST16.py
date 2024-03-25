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


class NIST16(AbstractDataset):
    """
    directory structure:
    NIST16
    │── img
    └── mask => have the same name as images
    """
    def __init__(self, opt, transform_list, tamp_list: str, mode: str):
        """
        :param crop_size: (H,W) or None
        :param tamp_list: '/root/MVSS-Net-master/data/NIST16.txt'
        """
        super(NIST16, self).__init__(crop_size=opt['datasets']['train']['GT_size'])
        self.transform_list = transform_list
        self._root_path = Path("/hotdata")
        self.mode = mode

        with open(tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        while True:
            try:
                assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
                tamp_path = self._root_path / self.tamp_list[index][0]
                mask_path = self._root_path / self.tamp_list[index][1]

                tensors = self._create_tensor(tamp_path, mask_path, self.transform_list, self.mode)
                break
            except Exception as e:
                print(f"error loading img {tamp_path} mask {mask_path}. please check!")
                index = np.random.randint(0,len(self)-1)
    
        return tensors


if __name__ == '__main__':
    root = dataset_root = Path("/data")

    imlist = []  
    # NIST16
    tamp_root = root / "nist16/img"
    mask_root = root / "nist16/mask"
    for file in os.listdir(tamp_root):
        imlist.append(','.join([str(Path("nist16/img") / file),
                            str(Path("nist16/mask") / file)]))
        assert (mask_root / file).is_file()

    print(len(imlist))  # 564

    # mask validation
    new_imlist=[]
    for s in imlist:
        im, mask = s.split(',')
        im_im = cv2.imread(str(root / im)) / 255.0
        mask_im = cv2.imread(str(root / mask), cv2.IMREAD_GRAYSCALE) / 255.0
        mask_im[mask_im > 0] = 1
        mask_im = 1 - mask_im
        if im_im.shape[0] != mask_im.shape[0] or im_im.shape[1] != mask_im.shape[1] or mask_im.sum() * 2 > mask_im.shape[0] * mask_im.shape[1]:
            print("Skip:", im, mask)
            continue
        new_imlist.append(s)

    with open("/root/DistillationForgertDetection/Distillation_model/Unet_with_MAE_Augmentation/datasets/NIST16.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 550

    
