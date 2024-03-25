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
import data.util as util

class CASIA2(AbstractDataset):
    """
    directory structure:
    CASIA2
    │── Au
    │── Tp
    └── Groundtruth  => Run renaming script in the excel file located in the above repo.
                            Plus, rename "Tp_D_NRD_S_N_cha10002_cha10001_20094_gt3.png" to "..._gt.png"
    """
    def __init__(self, opt, transform_list, tamp_list: str, mode: str):
        super(CASIA2, self).__init__(crop_size=opt['datasets']['train']['GT_size'])
        """
        :param crop_size: (H,W) or None
        :param tamp_list: '/root/MVSS-Net-master/data/CASIA2.txt'
        """
        self.transform_list = transform_list
        self._root_path = Path("/hotdata")
        self.mode = mode

        self.Au_images, _ = util.get_image_paths("/hotdata/CASIA2/Au")
        # self.Au_images, _ = util.get_image_paths("/hotdata/COCOdataset/train2017")

        # self.mask_sam_images_tp, _ = util.get_image_paths("/hotdata/CASIA2/Tp_SAM")
        # self.mask_sam_images_au, _ = util.get_image_paths("/hotdata/CASIA2/Au_SAM")
        # self.mask_sam = self.mask_sam_images_tp + self.mask_sam_images_au

        print(f"len of Au images {len(self.Au_images)}")


        with open(tamp_list, "r") as f:
            self.tamp_list = [t.strip().split(',') for t in f.readlines()]

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        try:
            tamp_path = self._root_path / self.tamp_list[index][0]
            mask_path = self._root_path / self.tamp_list[index][1]
            mask_path = str(mask_path).split('/')[1:]
            # mask_path[2] = 'CASIA 2 Groundtruth'
            mask_path = '/'+'/'.join(mask_path)
            
        except Exception as e:
            print(f"load image failure: {str(mask_path)}")
        return self._create_tensor(tamp_path, mask_path, self.transform_list, self.mode, with_au=True)

    def __len__(self):
        return len(self.tamp_list)


if __name__ == '__main__':
    root = dataset_root = Path("/data")

    imlist = []  
    # CASIA2
    tamp_root = root / "CASIA2/Tp"
    mask_root = root / "CASIA2/Groundtruth"
    for file in os.listdir(tamp_root):
        if file in ['Tp_D_NRD_S_B_ani20002_nat20042_02437.tif']:
            continue  # stupid file
        # In CASIA2, there are only two types of images: .jpg and .tif
        if not file.lower().endswith(".jpg"): 
            if not file.lower().endswith(".tif"):
                print(file)
                continue
            
        imlist.append(','.join([str(Path("CASIA2/Tp") / file),
                            str(Path("CASIA2/Groundtruth") / (os.path.splitext(file)[0] + "_gt.png"))]))

        assert (mask_root/(os.path.splitext(file)[0]+"_gt.png")).is_file()
    print(len(imlist))  # 5122

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

    with open("/root/DistillationForgertDetection/Distillation_model/Unet_with_MAE_Augmentation/datasets/CASIA2.txt", "w") as f:
        f.write('\n'.join(new_imlist)+'\n')
    print(len(new_imlist))  # 4975