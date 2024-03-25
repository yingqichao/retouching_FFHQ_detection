import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy
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
import albumentations as A
import data.util as util
import cv2

## todo: megvii和tencent读取上的区别: single上megvii不区分程度

class meiyanFFHQDataset(Dataset):
    def __init__(self, opt, is_train, dataset="megvii", with_origin=False):
        # 图片地址(先划分训练集和测试集)
        # self.with_dual = with_dual
        self.is_train = is_train
        self.dataset = dataset
        self.opt = opt
        ## todo: note-origin is only used to calculate psnr distribution of the dataset!
        self.with_origin = with_origin

        if self.dataset=="tencent":
            original_dir = '/hotdata/FFHQ'
            single_augment_dir = '/hotdata/FFHQ_process'
            dual_augment_dir = '/hotdata/FFHQ_dual_process'
            three_augment_dir = '/hotdata/FFHQ_three_process'
            four_augment_dir = '/hotdata/FFHQ_four_process'
        elif self.dataset=="megvii":
            original_dir = '/hotdata/FFHQ'
            single_augment_dir = '/hotdata/FFHQ_megvii_process'
            dual_augment_dir = '/hotdata/FFHQ_megvii_dual_process'
            three_augment_dir = '/hotdata/FFHQ_megvii_three_process'
            four_augment_dir = '/hotdata/FFHQ_megvii_four_process'
        elif self.dataset=="alibaba":
            ## todo: treat alibaba just as tencent, and accordingly change setting
            self.dataset = "tencent"
            original_dir = '/hotdata/FFHQ'
            single_augment_dir = '/hotdata/FFHQ_ali_process'
            self.is_train = "all"
            self.opt['test_operation_list'] = ['single']

        else:
            raise NotImplementedError("Dataset error, please check.")
        magnitude_split = 30
        ## todo: ban image list
        file_names = ['00_34_closingeye.txt',
                      'eye_close.txt',
                      'sunglasses.txt',
                      '00_34_baby.txt',
                      'child_35.txt'
                      ]
        self.ban_set = set()

        for file_name in file_names:
            with open(os.path.join('/hotdata/FFHQ_settings/excluded_images_list', file_name), 'r') as f:
                # ban_list += [line.strip() for line in f.readlines()]
                for line in f.readlines():
                    self.ban_set.add(line.strip())

        print(f"Ban image list: {len(self.ban_set)}")


        ## todo: downsample classes for balanced training
        if self.is_train=="test":
            self.skip_rate = [0., 0., 0., 0., 0.]
        else:
            if self.dataset == "tencent":
                self.skip_rate=[6/7., 2/3., 0., 0., 0.]
            else: ## which is megvii
                self.skip_rate = [5/7., 0., 0., 0., 0.]
        ## todo: categories
        self.category = ['EyeEnlarging', 'FaceLifting', 'Smoothing', 'Whitening']
        self.labels_codebook = {}
        ## todo: how many operations should be loaded?
        assert 'operations_list' in self.opt, "No operations_list in opt! check!"
        if "test_operation_list" not in self.opt:
            print("test_operation_list not found. using operation_list instead.")
            self.opt['test_operation_list'] = self.opt['operations_list']

        self.operations_list = self.opt['operations_list'] if self.is_train=="train" else self.opt['test_operation_list']
        images_dirs,image_label_files = [], []
        if 'none' in self.operations_list:
            images_dirs.append(original_dir)
            image_label_files.append("none")
        if 'single' in self.operations_list:
            images_dirs.append(single_augment_dir)
            image_label_files.append("single" if self.dataset=="tencent" else os.path.join(single_augment_dir,"one_process.txt"))
        if 'dual' in self.operations_list:
            images_dirs.append(dual_augment_dir)
            image_label_files.append(os.path.join(dual_augment_dir,"dual_process.txt"))
        if 'three' in self.operations_list:
            images_dirs.append(three_augment_dir)
            image_label_files.append(os.path.join(three_augment_dir,"three_process.txt"))
        if 'four' in self.operations_list:
            images_dirs.append(four_augment_dir)
            image_label_files.append(os.path.join(four_augment_dir,"four_process.txt"))
        if len(images_dirs)==0:
            raise NotImplementedError("No valid operation found for meiyandataset! check!")

        # assert isinstance(images_dir, str), "we currently only support loading one dataset."
        imgs = [] #, labels, references, magnitudes = [], [], [], []
        for index, images_dir in enumerate(images_dirs):
            label_file = image_label_files[index]
            category_dirs = [""] if label_file=='none' else sorted(os.listdir(images_dir))

            ## todo: if more than one operation, load text for label
            if ('single' == label_file and self.dataset!='megvii') or 'none' == label_file:
                pass
            else:
                print(f"from {label_file} loading files...")
                with open(label_file, "r") as f:
                    for line in f:
                        line = line.strip('\n').split('\t')
                        assert len(line)==2, f"strange! {line} len of label exceed 2, check!"
                        item_dict = eval(line[0])
                        item_label = [0,0,0,0]
                        for key in item_dict:
                            item_label[self.category.index(key)] = int(max(0,min(3, int(item_dict[key]) // magnitude_split)))
                        ## todo: if address contains groupshare, change to hotdata
                        address = line[1] #.replace("\\",'/')
                        address_index = address.rfind('/')
                        file_index = address[:address_index].rfind('/')
                        address_index = address[:file_index].rfind('/')
                        address_index = address[:address_index].rfind('/')
                        address = address[address_index+1:]
                        # if 'groupshare' in address:
                        #     address = '/hotdata'+address[11:]
                        self.labels_codebook['/hotdata/'+address] = item_label

            ## todo: load images and labels
            for idx, process_name in enumerate(category_dirs):
                ## todo: note: the first dir is original images
                main_folder = original_dir if label_file=='none' else images_dir
                process_dir = os.path.join(main_folder, process_name)
                if not os.path.isdir(process_dir):
                    continue
                oneK_dirs = sorted(os.listdir(process_dir))
                ## todo: train-val-test split
                index_train_end = int(len(oneK_dirs)*0.8)
                index_val_end = int(len(oneK_dirs) * 0.9)
                ## todo: take special care on original images! we should not leak the original images of the test images.
                if label_file=='none':
                    new_oneK_dirs, idx_process_dir = [], [0,20,40,60,70]
                    for iii in range(4):
                        if self.is_train == "train":
                            new_oneK_dirs = new_oneK_dirs + \
                                        oneK_dirs[idx_process_dir[iii]:int(idx_process_dir[iii]+(idx_process_dir[iii+1]-idx_process_dir[iii])*0.8)]
                        elif self.is_train == "val":
                            new_oneK_dirs = new_oneK_dirs + \
                                        oneK_dirs[idx_process_dir[iii]+int((idx_process_dir[iii+1]-idx_process_dir[iii])*0.8):
                                                  idx_process_dir[iii]+int((idx_process_dir[iii+1]-idx_process_dir[iii])*0.9)]
                        elif self.is_train == "test":
                            new_oneK_dirs = new_oneK_dirs + \
                                        oneK_dirs[idx_process_dir[iii]+int((idx_process_dir[iii+1]-idx_process_dir[iii])*0.9):idx_process_dir[iii+1]]
                        else: # "sum"
                            new_oneK_dirs = new_oneK_dirs + oneK_dirs[idx_process_dir[iii]:idx_process_dir[iii+1]]

                    oneK_dirs = new_oneK_dirs
                else:
                    if self.is_train == "train":
                        oneK_dirs = oneK_dirs[:index_train_end]
                    elif self.is_train == "val":
                        oneK_dirs = oneK_dirs[index_train_end:index_val_end]
                    elif self.is_train == "test":
                        oneK_dirs = oneK_dirs[index_val_end:]
                    else: # sum
                        pass

                for last_dir in oneK_dirs:
                    specific_dir = os.path.join(process_dir, last_dir)
                    if not os.path.isdir(specific_dir):
                        continue
                    ## todo: load image
                    tmp_imgs = []
                    for img_train in os.listdir(specific_dir):
                        if img_train[:5] not in self.ban_set:
                            tmp_imgs.append(os.path.join(specific_dir, img_train))

                    # tmp_imgs = [os.path.join(specific_dir, img_train) for img_train in
                    #             os.listdir(specific_dir)]
                    imgs += tmp_imgs
                    print(f"{is_train} Stage:{specific_dir} Added samples: {len(tmp_imgs)} Current total Samples:{len(imgs)}")
                    ## todo: load label and magnitude: for single/none operation, add magnitudes to the codebook
                    label = [0] * len(self.category)
                    magnitude = [0] * len(self.category)
                    if ('single' == label_file and self.dataset=="tencent") or 'none' == label_file:
                        if 'single' == label_file:
                            class_name = re.sub(r'[0-9_]+', '', process_name)
                            mag_name = re.sub(r'[A-Za-z_]+', '', process_name)
                            class_index = self.category.index(class_name)
                            mag_index = max(1,min(3, int(mag_name) // magnitude_split))
                            label[class_index] += 1
                            magnitude[class_index] += mag_index
                        # labels += [label for i in range(len(tmp_imgs))]
                        # magnitudes += [magnitude for i in range(len(tmp_imgs))]
                        for tmp_img in tmp_imgs:
                            self.labels_codebook[tmp_img] = magnitude

        self.transform_train = A.Compose(
            [
                # A.HorizontalFlip(p=0.5),
                A.Resize(always_apply=True,
                         height=self.opt['datasets']['train']['GT_size'],
                         width=self.opt['datasets']['train']['GT_size']),
                A.ImageCompression(always_apply=False, quality_lower=80, quality_upper=100, p=0.5),
                A.MotionBlur(p=0.5),
            ]
        )

        self.transform_test = A.Compose(
            [
                A.Resize(always_apply=True,
                         height=self.opt['datasets']['train']['GT_size'],
                         width=self.opt['datasets']['train']['GT_size']),
                A.MotionBlur(p=0.5),
                A.ImageCompression(always_apply=False, quality_lower=80, quality_upper=95, p=0.5),
            ]
        )

        self.transform_just_resize = A.Compose(
            [
                A.Resize(always_apply=True,
                         height=self.opt['datasets']['train']['GT_size'],
                         width=self.opt['datasets']['train']['GT_size'])
            ]
        )

        # self.train_transforms = transforms.Compose(
        #     [
        #         transforms.Resize((self.opt['datasets']['train']['GT_size'], self.opt['datasets']['train']['GT_size'])),
        #         # transforms.RandomResizedCrop(self.opt['datasets']['train']['GT_size']),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #     ]
        # )
        #
        # self.test_transforms = transforms.Compose(
        #     [
        #         transforms.Resize((self.opt['datasets']['train']['GT_size'], self.opt['datasets']['train']['GT_size'])),
        #         # transforms.CenterCrop(self.opt['datasets']['train']['GT_size']),
        #         transforms.ToTensor(),
        #     ]
        # )
        self.file_list = imgs  # train_list if is_train else test_list
        # self.file_labels = labels  # train_labels if is_train else test_labels
        # self.file_reference = references  # train_reference if is_train else test_reference
        # self.file_magnitude = magnitudes
        # self.transform = self.train_transforms if is_train else self.test_transforms

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        while True:
            GT_path = self.file_list[idx]

            ## todo: load labels and magnitudes
            # if 'single' in self.operations_list:
            #     label = self.file_labels[idx]
            #     degree = self.file_magnitude[idx]
            # else:
            if GT_path not in self.labels_codebook:
                print(f"[NotFoundWarning] {GT_path}")
                idx = np.random.randint(0,len(self.file_list))
                continue
            degree = self.labels_codebook[GT_path]
            label = [int(min(item, 1)) for item in degree]
            class_type = sum(label)
            ## todo: skip_rate: [6/7., 2/3., 0., 0., 0.]
            if self.is_train!="train" or np.random.random()>self.skip_rate[class_type]:
                break

        # img = Image.open(img_path)
        # img_transformed = self.transform(img)
        ## todo: load image
        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]

        ## todo: data augmentation
        if self.opt["with_data_aug"] and self.is_train!="sum":
            data_aug = self.transform_train if self.is_train=="train" else self.transform_test
        else:
            data_aug = self.transform_just_resize
        img_GT = data_aug(image=copy.deepcopy(img_GT))["image"]

        img_GT = img_GT.astype(np.float32) / 255.
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        # some images have 4 channels
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            # img_jpeg_GT = img_jpeg_GT[:, :, [2, 1, 0]]
            # img_LQ = img_LQ[:, :, [2, 1, 0]]

        ## todo: convert to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()

        ## todo: load origin images if needed
        if self.with_origin:
            slash_index = GT_path.rfind('1_')

            address_index = GT_path.rfind('/')
            file_index = GT_path[:address_index].rfind('/')
            if slash_index==-1:
                origin_path = '/hotdata/FFHQ/'+GT_path[file_index+1:]
            else:
                ## todo: some image contain x_x
                origin_path = '/hotdata/FFHQ/' + GT_path[file_index + 1:address_index]+"/"+GT_path[slash_index+2:]
            img_origin = cv2.imread(origin_path, cv2.IMREAD_COLOR)
            img_origin = util.channel_convert(img_origin.shape[2], 'RGB', [img_origin])[0]

            data_aug = self.transform_just_resize
            img_origin = data_aug(image=copy.deepcopy(img_origin))["image"]

            img_origin = img_origin.astype(np.float32) / 255.
            if img_origin.ndim == 2:
                img_origin = np.expand_dims(img_origin, axis=2)
            # some images have 4 channels
            if img_origin.shape[2] > 3:
                img_origin = img_origin[:, :, :3]

            # BGR to RGB, HWC to CHW, numpy to tensor
            if img_origin.shape[2] == 3:
                img_origin = img_origin[:, :, [2, 1, 0]]

            ## todo: convert to tensor
            img_origin = torch.from_numpy(np.ascontiguousarray(np.transpose(img_origin, (2, 0, 1)))).float()

            return img_GT, img_origin, label, degree, GT_path, origin_path
        else:
            return img_GT, label, degree, GT_path