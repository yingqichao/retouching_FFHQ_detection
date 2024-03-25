import argparse
import os
import random
import time

import numpy as np
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
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
from data.dataset_definition import get_dataset_and_loader
from sklearn.model_selection import train_test_split


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # torch.cuda._initialized = True
    # torch.backends.cudnn.benchmark = True
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print("world: {},rank: {},num_gpus:{}".format(world_size,rank,num_gpus))
    return world_size, rank

# seed = 123
def seed_everything(seed, rank):
    seed = seed*(rank+1)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def summary(logs, running_list, idx, valid_idx, print_step, epoch, model, rank, start, restart_step):
    for key in logs:
        if key not in running_list:
            running_list[key] = 0
        running_list[key] += logs[key]
    valid_idx += 1

    if valid_idx > 0 and (idx < 10 or valid_idx % print_step == print_step - 1):
        end = time.time()
        lr = logs['lr']
        info_str = f'[{epoch + 1}, {valid_idx + 1} {idx * model.data.shape[0]} {rank} {lr}] '
        for key in running_list:
            if "Num_" in key:
                ## todo: 表示数量的直接跳过
                pass
            # elif "TP" in key or "TN" in key:
            #     info_str += f'{key}: {running_list[key] / valid_idx:.3f} '
            else:
                count = running_list[f"Num_{key}"] if f"Num_{key}" in running_list else valid_idx
                info_str += f'{key}: {running_list[key] / (1e-6 + count):.3f} '
        info_str += f'time per sample {(end - start) / print_step / model.data.shape[0]:.3f} s'
        print(info_str)
        start = time.time()
        ## refresh the counter to see if the models behaves abnormaly.
        if valid_idx >= restart_step:
            running_list = {}  # [0.0] * len(variables_list)
            valid_idx = 0

    return start, running_list, valid_idx

def test(*, epoch, model, val_loaders: list, args, opt, rank, early_exit=None):
    print_step = 20
    current_step, valid_idx = 0, 0
    start = time.time()
    running_list, print_step, restart_step = {}, 100, 1000000
    ## todo: testing
    # model.test_ViT(epoch_number=epoch)
    for idx_testset, val_loader in enumerate(val_loaders):
        print(f"Test set: {val_loader.name_dataset}")
        is_last_testset = idx_testset == len(val_loaders) - 1
        for idx, batch in enumerate(val_loader):
            is_last = idx == len(val_loader) - 1
            model.feed_data_router_val(batch=batch, mode=args.mode)

            logs = model.test_ViT(epoch=epoch, step=current_step,
                                  index_info={
                                      'is_last': is_last,
                                      'is_last_testset': is_last_testset,
                                      'begin_10': idx <= 3,
                                  },
                                  running_stat=running_list,
                                  test_dataset_name=val_loader.name_dataset,
                                  )

            ## todo: counter
            start, running_list, valid_idx = summary(logs, running_list, idx, valid_idx, print_step, epoch, model, rank,
                                                     start, restart_step)

            current_step += 1

            if early_exit is not None and idx>=early_exit:
                print(f"End Test {val_loader.name_dataset}")
                break
    print("End Test")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('-task_name', type=str, default='meiyan')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-mode', type=int, default=0, help='validate or not.')
    # parser.add_argument('-task_name', type=str, default="COCO_base", help='will determine the name of the folder.')
    # parser.add_argument('-loading_from', type=str, default="COCO_base", help='loading checkpoints from?')
    # parser.add_argument('-load_models', type=int, default=1, help='load checkpoint or not.')
    args = parser.parse_args()
    opt = option.parse(opt_path=args.opt,
                       args=args)

    print('Enables distributed training.')
    world_size, rank = init_dist()

    seed = int(time.time())%1000
    print('Random seed: {}'.format(seed))
    # util.set_random_seed(seed)
    ## Slower but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    ## Faster but less reproducible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed, rank)

    ## todo: dataset definition
    train_set, val_set, train_sampler, train_loader, val_loaders = get_dataset_and_loader(opt=opt,args=args,
                                                                                         rank=rank,seed=seed)

    ## todo: models definition
    from models.model_definition import model_definition
    model = model_definition(args=args, opt=opt, train_loader=train_loader, val_loader=val_loaders)

    ## todo: training/testing script

    current_step, valid_idx = 0, 0
    start = time.time()
    epochs = 50 if opt['conduct_train'] else 1
    test_count = 0

    for epoch in range(epochs):
        running_list, print_step, restart_step = {}, 200, opt['restart_step']
        if opt['conduct_train']:
            print(f"Staring epoch {epoch}...")
            for idx, batch in enumerate(train_loader):
                # test_count = (current_step * batch[0].shape[0]) % opt['conduct_test_inverval_samples']
                test_count += (batch[0].shape[0])
                ## todo: in-batch early test
                if opt["conduct_in_batch_test"] and test_count >= opt['conduct_test_inverval_samples']:
                    test_count = 0
                    print(f"Staring testing epoch {epoch}...")
                    test(epoch=epoch, model=model, val_loaders=val_loaders, args=args,
                         opt=opt, rank=rank)


                model.feed_data_router(batch=batch, mode=args.mode)

                logs = model.train_ViT(
                                        epoch=epoch,
                                       index_info={
                                           'is_last': idx == len(train_loader) - 1,
                                           'begin_10': idx <= 3,
                                       },
                                       step=current_step)

                ## todo: counter
                start, running_list, valid_idx = summary(logs, running_list, idx, valid_idx, print_step, epoch, model,
                                                         rank, start, restart_step)

                current_step += 1

        ## todo: post-batch full-test
        if opt["conduct_test"]:
            print(f"Staring testing epoch {epoch}...")
            test(epoch=epoch, model=model, val_loaders=val_loaders, args=args,
                 opt=opt, rank=rank)


if __name__ == '__main__':
    main()

