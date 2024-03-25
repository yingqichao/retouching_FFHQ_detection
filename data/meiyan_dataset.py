import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import random
import os
import re
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class meiyanDataset(Dataset):
    def __init__(self, opt, is_train, with_reference, with_dual=False):
        # 图片地址(先划分训练集和测试集)
        self.with_reference = with_reference
        self.with_dual = with_dual
        self.is_train = is_train
        dir = '/groupshare/meiyan_calibrate/train' if is_train else '/groupshare/meiyan_calibrate/test'
        dual_dir = '/groupshare/meiyan_calibrate/meiyan_reprocess/train'
        # test_dir = '/groupshare/meiyan_calibrate/test'
        subdirs = sorted(os.listdir(dir))
        self.category = list(set([re.sub(r'[0-9]+', '', item) for item in subdirs]))
        # self.degree = ['30','60','90']
        imgs = []
        # test_list = []
        labels = []
        # test_labels = []
        ## 原始reference单独保存：(注意，有的时候reference可能被分到test里面去了)
        references = []
        # test_reference = []
        ## Degree
        magnitudes = []
        # test_magnitude = []

        # train = sum[:0.85*total]
        # test = sum[0.85*total:]

        for process_label in subdirs:
            tmp_imgs = [os.path.join(dir, process_label, img_train) for img_train in
                                   os.listdir(os.path.join(dir, process_label))]
            imgs += tmp_imgs
            ## todo: reference means including original photo during training to augment difference
            if self.with_reference:
                train_process_reference = [os.path.join(dir, "non-makeup", img_train) for img_train in
                                       os.listdir(os.path.join(dir, process_label))]
                references += train_process_reference
            ## todo: dual means process the image
            if self.with_dual:
                train_process_reference = [os.path.join(dir, "non-makeup", img_train) for img_train in
                                       os.listdir(os.path.join(dir, process_label))]
                references += train_process_reference

            class_name = re.sub(r'[0-9]+', '', process_label)
            index = self.category.index(class_name)
            labels += [index for i in range(len(tmp_imgs))]

            class_name = re.sub(r'[A-Za-z_]+', '', process_label)
            if len(class_name)==0:
                index = 0
            # index = 0 if len(class_name)==0 else int(class_name)/100
            else:
                index = min(2,int(class_name)//33)
            magnitudes += [index for i in range(len(tmp_imgs))]

            # test_process_label = [os.path.join(test_dir, process_label, img_test) for img_test in
            #                       os.listdir(os.path.join(test_dir, process_label))]
            # test_list += test_process_label
            # if self.with_reference:
            #     test_process_reference = [os.path.join(test_dir, "non-makeup", img_test) for img_test in
            #                       os.listdir(os.path.join(test_dir, process_label))]
            #     test_reference += test_process_reference
            # test_labels += [process_label for i in range(len(test_process_label))]

        print(f"{'train' if is_train else 'test' } Samples:{len(imgs)}")

        # todo: random show images
        # random_idx = np.random.randint(1, len(train_list), size=9)
        # fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        #
        # for idx, ax in enumerate(axes.ravel()):
        #     img = Image.open(train_list[random_idx[idx]])
        #     ax.set_title(train_labels[random_idx[idx]])
        #     ax.imshow(img)

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((opt['datasets']['train']['GT_size'], opt['datasets']['train']['GT_size'])),
                # transforms.RandomResizedCrop(opt['datasets']['train']['GT_size']),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((opt['datasets']['train']['GT_size'], opt['datasets']['train']['GT_size'])),
                # transforms.CenterCrop(opt['datasets']['train']['GT_size']),
                transforms.ToTensor(),
            ]
        )
        self.file_list = imgs #train_list if is_train else test_list
        self.file_labels = labels #train_labels if is_train else test_labels
        self.file_reference = references #train_reference if is_train else test_reference
        self.file_magnitude = magnitudes
        self.transform = self.train_transforms if is_train else self.test_transforms

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        # 标签似乎要变成数值?先用一维试试（不知道怎么检查）
        label = self.file_labels[idx]
        degree = self.file_magnitude[idx]
        # category = ['non-makeup', 'eyeEnlarging30', 'faceLifting30', 'whiting30', 'eyeEnlarging60', 'faceLifting60',
        #             'whiting60', 'eyeEnlarging90', 'faceLifting90', 'whiting90', 'synthesize50']
        # label = #self.category.index(label_expression)

        if not self.with_reference:
            return img_transformed, label, degree
        else:
            img_path = self.file_reference[idx]
            if not os.path.exists(img_path):
                pathlist = img_path.split('/')
                pathlist[-3] = 'test' if self.is_train else 'train'
                img_path = '/'+os.path.join(*pathlist)

            img = Image.open(img_path)
            img_transformed_reference = self.transform(img)
            return img_transformed, img_transformed_reference, label, degree