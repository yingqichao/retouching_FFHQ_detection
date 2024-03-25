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

from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['models']  # 提前网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
    for parameter in model.parameters():
        parameter.requires_grad = False  # 固定权重参数不变（因为仅用于测试？）
    model.eval()  # 此时模型仅用于测试，如果要训练改成.train()

    result = {'epoch_acc_list': checkpoint['epoch_acc_list'],
              'epoch_loss_list': checkpoint['epoch_loss_list'],
              'epochs': checkpoint['epochs']}
    return model, result