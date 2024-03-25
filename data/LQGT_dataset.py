import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
# from turbojpeg import TurboJPEG
from PIL import Image
# from jpeg2dct.numpy import load, loads
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import albumentations as A
import copy

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt, args, is_train=True):
        super(LQGTDataset, self).__init__()
        self.is_train = is_train
        self.opt = opt
        self.args = args
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.GT_size = opt['datasets']['train']['GT_size']
        self.name_dataset = "COCO"

        self.Au_paths, _ = util.get_image_paths("/hotdata/COCOdataset/val_2017/val2017_SAM") #("/hotdata/CASIA2/Au_SAM")
        self.Tp_paths, _ = util.get_image_paths("/hotdata/CASIA2/Tp_SAM")
        ## todo: additional forgery path
        self.COCO_paths, _ = util.get_image_paths("/hotdata/COCOdataset/train2017/train2017" if is_train else
                                                  "/hotdata/COCOdataset/val2017/val2017")

        self.transform_train = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.Resize(always_apply=True,
                         height=self.GT_size,
                         width=self.GT_size),
                # A.ImageCompression(always_apply=True, quality_lower=80, quality_upper=100, ),
                # A.MotionBlur(always_apply=True,),
            ]
        )

        # self.transform_test = A.Compose(
        #     [
        #         A.Resize(always_apply=True,
        #                  height=self.GT_size,
        #                  width=self.GT_size),
        #         A.MotionBlur(p=0.5),
        #         A.ImageCompression(always_apply=False, quality_lower=80, quality_upper=95, p=0.5),
        #     ]
        # )

        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True,
                         height=self.GT_size,
                         width=self.GT_size)
            ]
        )

        # 获取卷积核
        # self.kernel_erode = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(4, 4))
        # self.kernel_dilate = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(4, 4))

        # assert self.paths_GT, 'Error: GT path is empty.'

        # self.random_scale_list = [1]

        # self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')


    def __getitem__(self, index):

        # scale = self.dataset_opt['scale']

        # get GT image
        while True:
            mask_path = self.Au_paths[index-len(self.Tp_paths)] if index>=len(self.Tp_paths) else self.Tp_paths[index]
            mask = self.read_a_image(mask_path, mode=cv2.IMREAD_GRAYSCALE)
            lower_bool = torch.mean(mask)>0.04
            upper_bool = True
            if self.args.task_name == "imuge":
                upper_bool = torch.mean(mask) < 0.25

            if lower_bool and upper_bool:
                break
            else:
                index = np.random.randint(0, len(self.Au_paths)+len(self.Tp_paths)-1)

        image_index = np.random.randint(0, len(self.COCO_paths)-1)
        GT_path = self.COCO_paths[image_index]
        img_GT = self.read_a_image(GT_path, mode=cv2.IMREAD_COLOR)


        # orig_height, orig_width, _ = img_GT.shape
        # H, W, _ = img_GT.shape
        #
        # img_gray = rgb2gray(img_GT)
        # sigma = 2 #random.randint(1, 4)
        #
        #
        # canny_img = canny(img_gray, sigma=sigma, mask=None)
        # canny_img = canny_img.astype(np.float)
        # canny_img = self.to_tensor(canny_img)

        # canny_img = torch.from_numpy(np.ascontiguousarray(canny_img)).float()
        # img_jpeg_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_jpeg_GT, (2, 0, 1)))).float()
        # img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # if LQ_path is None:
        #     LQ_path = GT_path

        return img_GT, mask

    def read_a_image(self, GT_path, mode):
        # img_GT = util.read_img(GT_path)
        img_GT = cv2.imread(GT_path, mode)
        # img_GT = util.channel_convert(img_GT.shape[2], self.dataset_opt['color'], [img_GT])[0]

        if mode==cv2.IMREAD_GRAYSCALE or not self.is_train:
            img_GT = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        else:
            img_GT = self.transform_train(image=copy.deepcopy(img_GT))["image"]

        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        return img_GT

    def __len__(self):
        return len(self.Au_paths)+len(self.Tp_paths)

    # def to_tensor(self, img):
    #     img = Image.fromarray(img)
    #     img_t = F.to_tensor(img).float()
    #     return img_t
