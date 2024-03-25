from data.meiyan_ffhq_dataset import meiyanFFHQDataset as meiyanDataset
from models.networks import *
import math
from data.data_sampler import DistIterSampler
import torch.distributed as dist
from data.LQGT_dataset import LQGTDataset
# def get_dataset_and_loader(*, opt,args,rank,seed):
#     if args.task_name=="meiyan":
#         return get_dataset_and_loader_for_meiyan(opt=opt,args=args,rank=rank,seed=seed)
#     elif args.task_name=="forgery":
#         return get_dataset_and_loader_for_SAM_forgery_detection(opt=opt, args=args, rank=rank, seed=seed)
#     else:
#         raise NotImplementedError("加载dataloader失败，请检查！")

# def get_dataset_and_loader_for_SAM_forgery_detection(*, opt, args, rank, seed):
#     pass
#
# def get_dataset_and_loader_for_forgery_detection(*, opt, args, rank, seed):
#     from datasets.data_core import SplicingDataset
#     train_dataset = SplicingDataset(opt=opt, mode="train")
MODE_REAL = 0
MODE_SYNTHESIZE = 1

def get_dataset_and_loader(*, opt, args, rank, seed):
    dataset_opt = opt['datasets']['train']
    test_sets = []
    ## todo: dataset definition
    if args.task_name=="meiyan":
        train_set = meiyanDataset(opt=opt, is_train="train", dataset=opt['train_dataset'], with_origin=args.mode==1)
        val_set = meiyanDataset(opt=opt, is_train="val", dataset=opt['test_dataset'], with_origin=args.mode==1)
        test_set = meiyanDataset(opt=opt, is_train="test", dataset=opt['test_dataset'], with_origin=args.mode==1)
        test_sets.append(test_set)
    elif args.task_name=="forgery":
        from datasets.data_core import SplicingDataset
        import datasets.sync_transform as sync_transform
        train_transform_list = [
            sync_transform.DualTo_tensor(),
            sync_transform.DualRandomHorizonalFlip(),
            # sync_transform.DualRandomVerticalFlip(),
            sync_transform.DualResize(crop_size=[opt['datasets']['train']['GT_size'], opt['datasets']['train']['GT_size']]),
        ]
        test_transform_list = [
            sync_transform.DualTo_tensor(),
            sync_transform.DualResize(crop_size=[opt['datasets']['train']['GT_size'], opt['datasets']['train']['GT_size']]),
        ]
        if args.mode == MODE_SYNTHESIZE:
            ## todo: synthesized training set using COCO
            train_set = LQGTDataset(opt, args, is_train=True)
        else:
            ## todo: real training set (CASIA2)
            train_set = SplicingDataset(opt, transform_list=train_transform_list, mode="train", name_dataset="CASIA2")
        val_set = SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="CASIA1")
        # if args.mode ==MODE_SYNTHESIZE:
        test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="Columbia"))
        test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="IMD2020"))
        test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="NIST16"))
        test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="COVER"))
        test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="CASIA1"))
        # else:
        #     test_sets.append(SplicingDataset(opt, transform_list=test_transform_list, mode="valid", name_dataset="CASIA1"))
    elif args.task_name=="imuge":
        train_set = LQGTDataset(opt, args, is_train=True)
        val_set = LQGTDataset(opt, args, is_train=False)
        test_set = LQGTDataset(opt, args, is_train=False)
        test_sets.append(test_set)
    else:
        raise NotImplementedError("检查dataset名字")
    
    world_size = torch.distributed.get_world_size()
    print("World size: {}".format(world_size))
    print("Batch size: {}".format(dataset_opt['batch_size']))
    num_workers = dataset_opt['n_workers']
    assert dataset_opt['batch_size'] % world_size == 0
    batch_size = dataset_opt['batch_size'] // world_size

    dataset_ratio = dist.get_world_size() #200  # enlarge the size of each epoch
    # if opt['dist']:
    train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio, seed=seed)
    # else:
    #     train_sampler = None

    ## todo: train loader
    train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler, drop_last=True,
                                               pin_memory=True)

    print('Number of train images: {:,d}, iters: {:,d}'.format(
        len(train_set), train_size))

    ## todo: val loader
    val_size = int(math.ceil(len(val_set) / 1))
    print('Number of val images: {:,d}, iters: {:,d}'.format(
        len(val_set), val_size))

    ## todo: test loader
    # val_loader = create_dataloader(val_set, dataset_opt, opt, val_sampler)
    test_loaders = []
    for test_set in test_sets:
        test_size = int(math.ceil(len(test_set) / 1))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                 pin_memory=True, drop_last=False)

        print('Number of test images: {:,d}, iters: {:,d}'.format(
            len(test_set), test_size))

        test_loader.name_dataset = test_set.name_dataset
        test_loaders.append(test_loader)

    # print(len(train_data),len(train_labels),len(train_loader))
    # print(len(test_data),len(test_labels),len(test_loader))

    return train_set, test_sets, train_sampler, train_loader, test_loaders

    # elif args.mode==1:
    #     test_set = meiyanDataset(opt=opt, is_train="test", dataset="megvii", with_origin=True)
    #
    #     # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    #     # test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    #     world_size = torch.distributed.get_world_size()
    #     print("World size: {}".format(world_size))
    #     print("Batch size: {}".format(dataset_opt['batch_size']))
    #     num_workers = dataset_opt['n_workers']
    #     assert dataset_opt['batch_size'] % world_size == 0
    #     batch_size = dataset_opt['batch_size'] // world_size
    #
    #     dataset_ratio = dist.get_world_size()  # 200  # enlarge the size of each epoch
    #     # if opt['dist']:
    #     test_sampler = DistIterSampler(test_set, world_size, rank, dataset_ratio, seed=seed)
    #     # else:
    #     #     train_sampler = None
    #
    #
    #     train_size = int(math.ceil(len(test_set) / dataset_opt['batch_size']))
    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
    #                                                num_workers=num_workers, sampler=test_sampler, drop_last=True,
    #                                                pin_memory=True)
    #
    #     print('Number of train images: {:,d}, iters: {:,d}'.format(
    #         len(test_set), train_size))
    #
    #
    #     return None, test_set, None, None, test_loader
