import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from models.base_model import BaseModel
from models.networks import ViT
from torch.nn.parallel import DistributedDataParallel
from utils import stitch_images

# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"

class meiyan_multiscale(BaseModel):
    def __init__(self, opt, args, train_loader=None, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.args = args
        self.history_accuracy = 0.2 # minimum required accuracy

        self.methodology = opt['methodology']
        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(meiyan_multiscale, self).__init__(opt, args)
        ### todo: options
        from models.networks import UNetDiscriminator
        self.recovery_network = UNetDiscriminator().cuda()
        self.recovery_network = DistributedDataParallel(self.recovery_network,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)

        if "vit" in opt['network_arch']:
            from transformers import ViTConfig, ViTModel
            configuration = ViTConfig()
            self.detection_model = ViTModel(configuration).cuda()
            # self.detection_model = ViT(
            #     dim=1024,  # 1024
            #     image_size=224,  # 图片大小361*361，需要处理
            #     patch_size=16,
            #     num_classes=11,  # 类别多少？
            #     depth=6,  # 6
            #     heads=16,  # 16
            #     mlp_dim=2048,  # 2048
            #     dropout=0.1,
            #     emb_dropout=0.1
            # ).cuda()
        elif "resnet" in opt['network_arch']:
            from CNN_architectures.pytorch_resnet import ResNet50
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = ResNet50(img_channel=6,num_classes=11, use_SRM=False, feat_concat=True).cuda()
        elif "tres" in opt['network_arch']:
            from models.IFA.tres_model import Net
            self.detection_model = Net(num_classes=11).cuda()
        else:
            raise NotImplementedError("模型结构不对，请检查！")
        print(f"{opt['network_arch']} models created. method: {opt['methodology']}")
        self.detection_model = DistributedDataParallel(self.detection_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)

        if self.opt['model_load_number'] > 0:
            self.reload(
                pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{self.opt['model_load_number']}_ViT.pth",
                network=self.detection_model,
                strict=True)
            self.reload(
                pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{self.opt['model_load_number']}_recovery.pth",
                network=self.recovery_network,
                strict=True)

        self.criterion = nn.CrossEntropyLoss()
        self.train_opt = opt['train']

        # self.optimizer = Adam(self.detection_model.parameters(), lr=lr)
        wd_G = 1e-5
        self.optimizer = self.create_optimizer(self.detection_model,
                                               lr=self.train_opt['lr_scratch'], weight_decay=wd_G)
        self.optimizer_recovery = self.create_optimizer(self.recovery_network,
                                               lr=2*self.train_opt['lr_scratch'], weight_decay=wd_G)
        # scheduler
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=50000)
        self.schedulers.append(scheduler)
        scheduler = CosineAnnealingLR(self.optimizer_recovery, T_max=50000)
        self.schedulers.append(scheduler)

    def feed_data_router(self, *, batch, mode):
        ## 赛数据
        data, reference, label = batch
        self.data = data.cuda()
        self.reference = reference.cuda()
        self.label = label.cuda()

    def train_ViT(self, *, step=None):
        self.detection_model.train()
        self.recovery_network.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr


        ## todo: first step: reference recovery
        estimated_ft, mid_feats = self.recovery_network(self.data)
        loss_recovery = self.l1_loss(estimated_ft,self.reference)
        estimated_ft = self.clamp_with_grad(estimated_ft)
        PSNR = self.psnr(self.postprocess(estimated_ft),
                               self.postprocess(self.reference)).item()
        output = self.detection_model(torch.cat([self.data,estimated_ft-self.data],dim=1),
                                      mid_feats_from_recovery=mid_feats)
        loss_ce = self.criterion(output, self.label)

        loss = loss_ce + loss_recovery
        loss.backward()
        nn.utils.clip_grad_norm_(self.detection_model.parameters(), 1)
        nn.utils.clip_grad_norm_(self.recovery_network.parameters(), 1)
        self.optimizer.step()
        self.optimizer_recovery.step()
        self.optimizer.zero_grad()
        self.optimizer_recovery.zero_grad()

        acc = (output.argmax(dim=1) == self.label).float().mean()
        logs['epoch_accuracy'] = acc
        logs['loss_ce'] = loss_ce
        logs['loss_recovery'] = loss_recovery
        logs['PSNR'] = PSNR

        if (self.global_step % 1000 == 3 or self.global_step <= 10):
            images = stitch_images(
                self.postprocess(self.data),
                self.postprocess(self.reference),
                self.postprocess(estimated_ft),
                self.postprocess(10 * torch.abs(self.data - estimated_ft)),
                self.postprocess(10 * torch.abs(self.reference - estimated_ft)),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        self.global_step = self.global_step + 1

        return logs

    def test_ViT(self, *, epoch_number):
        ## todo: testing
        epoch_loss = 0
        epoch_accuracy = 0
        self.detection_model.eval()
        for idx, batch in enumerate(self.val_loader):
            data, reference, label = batch
            data = data.cuda()
            label = label.cuda()
            estimated_ft, mid_feats = self.recovery_network(data)
            output = self.detection_model(torch.cat([data, estimated_ft - data], dim=1),
                                          mid_feats_from_recovery=mid_feats)
            loss = self.criterion(output, label)

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(self.val_loader)
            epoch_loss += loss.item() / len(self.val_loader)
        print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} \n")

        if epoch_accuracy>self.history_accuracy:
            print(f'Saving models and training states.')
            # self.model_save(path='checkpoint/latest', epochs=self.global_step)
            self.save(accuracy=int(epoch_accuracy*100), iter_label=epoch_number)
            self.history_accuracy = epoch_accuracy

    def save(self, *, accuracy, iter_label):
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{accuracy}_{iter_label}_ViT.pth",
            network=self.detection_model,
        )

        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.args.task_name}/{accuracy}_{iter_label}_recovery.pth",
            network=self.detection_model,
        )

        # self.save_network(network=self.detection_model, network_label='ViT', accuracy=accuracy,
        #                   iter_label=iter_label if self.rank == 0 else 0,
        #                   model_path=self.out_space_storage + f'/{folder}/{self.task_name}/')


