import copy
import os.path
from abc import ABC, abstractmethod
from PIL import Image, JpegImagePlugin
import numpy as np
import math
# import jpegio  # See https://github.com/dwgoon/jpegio/blob/master/examples/jpegio_tutorial.ipynb
# import torch_dct as dct
import torch
import random
import cv2
import albumentations as A
import data.util as util

class AbstractDataset(ABC):
    YCbCr2RGB = torch.tensor([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]], dtype=torch.float64)

    def __init__(self, crop_size):
        """
        :param crop_size: (H, W) or None
        """
        self._crop_size = crop_size
        self.tamp_list = None

        self.transform_train = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.Resize(always_apply=True,
                         height=crop_size,
                         width=crop_size),
                A.ImageCompression(always_apply=False, quality_lower=80, quality_upper=100,p=0.5),
                # A.MotionBlur(always_apply=True,),
            ]
        )

        # self.transform_test = A.Compose(
        #     [
        #         A.Resize(always_apply=True,
        #                  height=crop_size,
        #                  width=crop_size),
        #         A.MotionBlur(p=0.5),
        #         A.ImageCompression(always_apply=False, quality_lower=80, quality_upper=95, p=0.5),
        #     ]
        # )

        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True,
                         height=crop_size,
                         width=crop_size)
            ]
        )

        # 获取卷积核
        self.kernel_erode = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(4, 4))
        self.kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(4, 4))


    def _create_tensor(self, im_path, msk_path, transform_list, mode, with_au=False):

        img_RGB = cv2.imread(str(im_path))
        img = img_RGB[:, :, [2, 1, 0]]
        # img_store = copy.deepcopy(img)

        if mode == "train":
            data_aug = self.transform_train
        else:
            data_aug = self.transform_just_resize

        img = data_aug(image=img)["image"]

        # img_store = data_aug(image=img_store)["image"]

        # dilate_ndarr = cv2.dilate(src=erode_ndarr, kernel=kernel_dilate, iterations=1)

        # # only use the image augmentation when training!
        # if mode == "train":
        #     if torch.rand(1) <= 0.5:
        #         img = cv2.GaussianBlur(img, (5, 5), 1.5)
        #     if torch.rand(1) <= 0.5:
        #         jpeg_quality = 90
        #         if str(im_path).split('.')=='jpg':
        #             img = cv2.imdecode(cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])[1], cv2.IMREAD_UNCHANGED)
        #         else:
        #             img = cv2.imdecode(cv2.imencode('.tif', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])[1], cv2.IMREAD_UNCHANGED)

        # img_store = img_store / 255.0

        ## todo: mask and edge generation
        
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self._crop_size, self._crop_size))
        mask = np.where(mask > 127.5, 255, 0)


        ## todo: loading authentic images
        # Au_path = self.Au_images[np.random.randint(0, len(self.Au_images)-1)]
        # img_Au = cv2.imread(str(Au_path))
        # img_Au = img_Au[:, :, [2, 1, 0]]
        # img_Au = data_aug(image=img_Au)["image"]

        # erode_ndarr = cv2.erode(src=copy.deepcopy(mask), kernel=self.kernel_erode, iterations=1)
        # dilate_ndarr = cv2.dilate(src=copy.deepcopy(mask), kernel=self.kernel_dilate, iterations=1)
        # img_store = dilate_ndarr - erode_ndarr
        # img_store = np.where(img_store > 127.5, 1, 0)

        ## todo: auth image
        # for t in transform_list:
        #     img, img_store, mask = t(img, img_store, mask)
        img = self.t(img)
        mask = self.t(mask)

        # ## todo: loading SAM models
        # paths = str(im_path).split('/')
        # paths[3] = paths[3] + "_SAM"
        # filename = paths[-1]
        # paths[-1] = filename[:filename.rfind('.')]
        # SAM_folder = "/".join(paths)
        # SAM_paths, _ = util.get_image_paths(SAM_folder)
        # img_Au = np.zeros((self._crop_size, self._crop_size, 320))
        # num_SAM = min(320, len(SAM_paths))
        # for idx in range(num_SAM):
        #     SAM_path = SAM_paths[idx]
        #     SAM_mask = cv2.imread(str(SAM_path), cv2.IMREAD_GRAYSCALE)
        #     SAM_mask = cv2.resize(SAM_mask, (self._crop_size, self._crop_size))
        #     SAM_mask = np.where(SAM_mask > 127.5, 255, 0)
        #     img_Au[:, :, 320 - 1 - idx] = SAM_mask
        #
        # img_store = self.t(img_Au)  # if with_au else img

        return img, img, mask

    def t(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # if img.shape[2] > 3:
        #     img = img[:, :, :3]
        img = img / 255.0
        # if img.shape[2] == 3:
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(np.ascontiguousarray(img)).float()

        return img

    @abstractmethod
    def get_tamp(self, index):
        pass

    def get_tamp_name(self, index):
        item = self.tamp_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

    def __len__(self):
        return len(self.tamp_list)

