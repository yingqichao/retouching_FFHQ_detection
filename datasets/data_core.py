import torch
from torch.utils.data import Dataset
import random
from .dataset_CASIA2 import CASIA2
from .dataset_IMD2020 import IMD2020
from .dataset_CASIA1 import CASIA1
from .dataset_COVER import COVER
from .dataset_NIST16 import NIST16
from datasets.Columbia_dataset import Columbia_dataset


class SplicingDataset(Dataset):
    def __init__(self, opt, transform_list, name_dataset, mode="train"):
        self.dataset_list = []
        # self.train_dataset="CASIA2"
        # self.test_dataset="NIST16"
        self.name_dataset = name_dataset #self.train_dataset if mode=="train" else self.test_dataset
        print(f"using {name_dataset} as {mode}")
        # if mode == "train":
        if "CASIA2" == name_dataset:
            self.dataset_list.append(CASIA2(opt, transform_list, "./datasets/CASIA2.txt", mode))
            # self.dataset_list.append(IMD2020(transform_list, "/root/DistillationForgertDetection/data/IMD2020.txt", mode))
            # self.dataset_list.append(NIST16(transform_list, "/root/DistillationForgertDetection/data/NIST16.txt", mode))
            # self.dataset_list.append(COVER(transform_list, "/root/DistillationForgertDetection/data/COVER.txt", mode))
            # self.dataset_list.append(CASIA1(transform_list, "/root/DistillationForgertDetection/data/CASIA1.txt", mode))
        # elif mode == "valid":
            # self.dataset_list.append(CASIA2(transform_list, "./datasets/CASIA2.txt", mode))
        elif "IMD2020" == name_dataset:
            self.dataset_list.append(IMD2020(opt, transform_list, "./data/IMD2020.txt", mode))
        elif "NIST16" == name_dataset:
            self.dataset_list.append(NIST16(opt, transform_list, "./datasets/NIST16.txt", mode))
        elif "COVER" == name_dataset:
            self.dataset_list.append(COVER(opt, transform_list, "./data/COVER.txt", mode))
        elif "CASIA1" == name_dataset:
            self.dataset_list.append(CASIA1(opt, transform_list, "./datasets/CASIA1.txt", mode))
        elif "Columbia" == name_dataset:
            self.dataset_list.append(Columbia_dataset(opt, True, mode))
        else:
            raise KeyError("Invalid dataset: " + name_dataset)
        self.tranform_list = transform_list
        self.mode = mode

    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.tamp_list)

    def get_filename(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp_name(index)

    def __len__(self):
        return sum([len(lst) for lst in self.dataset_list])

    def __getitem__(self, index):
        it = 0
        # while True:
        #     if index >= len(self.dataset_list[it]):
        #         index -= len(self.dataset_list[it])
        #         it += 1
        #         continue
        if self.name_dataset != "Columbia":
            return self.dataset_list[it].get_tamp(index)
        else:
            img, img, mask = self.dataset_list[it].__getitem__(index)
            return img, img, mask

    def get_info(self):
        s = ""
        for ds in self.dataset_list:
            s += (str(ds)+'('+str(len(ds))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size}\n"
        return s





