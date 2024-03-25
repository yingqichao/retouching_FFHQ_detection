import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from omegaconf import OmegaConf
from models.base_model import BaseModel
from models.networks import ViT
from torch.nn.parallel import DistributedDataParallel
from losses.focal_loss import focal_loss
from losses.dice_loss import DiceLoss
from utils import stitch_images
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
# from torchvision.ops.focal_loss import sigmoid_focal_loss
from losses.adaptive_focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
# from torchmetrics import F1
from torchmetrics.classification import BinaryF1Score
from losses.dice_loss import SmoothF1
import os
import yaml
import torchvision.transforms.functional as Functional
from skimage.feature import canny
from skimage.color import rgb2gray
import random
import cv2
from inpainting_methods.lama_models.my_own_elastic_dtcwt import my_own_elastic

class imuge_main(BaseModel):
    def __init__(self, opt, args, train_loader=None, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.args = args
        self.history_accuracy = 0.2
        self.methodology = opt['methodology']
        self.class_name = ["眼", "提", "平", "白"]

        self.MODE_REAL = 0
        self.MODE_SYNTHESIZE = 1

        self.F1 = BinaryF1Score().cuda()
        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(imuge_main, self).__init__(opt, args)
        classes = [4, 4, 4, 4]
        print(f"classes: {classes}")
        if 'focal_alpha' not in self.opt:
            print("focal_alpha not in opt. using default 0.5.")
            self.opt['focal_alpha'] = 0.5
        self.focal_loss = focal_loss(alpha=self.opt['focal_alpha'], gamma=1, num_classes=4).cuda()
        self.dice_loss = DiceLoss().cuda()
        print(f"current mode is {args.mode}")
        ### todo: 定义分类网络
        # if "ddpm_unet" in self.opt["network_arch"]:
        from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
        from models.networks import UNetDiscriminator
        # self.protection_model = UNetDiscriminator(out_channels=3).cuda()
        self.protection_model = Unet(dim=32, out_channels=3, use_SRM=False, residual_connection=True, apply_bottleneck=True).cuda()
        # self.detection_model = UNetDiscriminator(out_channels=1).cuda()
        self.detection_model = Unet(dim=32, out_channels=1, use_SRM=3, residual_connection=True, apply_bottleneck=True).cuda()
        # self.recovery_model = UNetDiscriminator(out_channels=3).cuda()
        self.recovery_model = Unet(dim=32, out_channels=3, use_SRM=False, residual_connection=True,apply_bottleneck=True).cuda()
        # else:
        #     raise NotImplementedError("没见过的model！")

        self.detection_model = DistributedDataParallel(self.detection_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)
        self.protection_model = DistributedDataParallel(self.protection_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)
        self.recovery_model = DistributedDataParallel(self.recovery_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)

        ## todo: 加载之前的模型
        if self.opt['model_load_number'] is not None:
            self.reload(pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/DETECT_{self.opt['model_load_number']}",
                        network=self.detection_model,
                        strict=False)
            self.reload(pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/PROTECT_{self.opt['model_load_number']}",
                        network=self.protection_model,
                        strict=False)
            self.reload(pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/RECOVERY_{self.opt['model_load_number']}",
                        network=self.recovery_model,
                        strict=False)

        self.criterion = nn.CrossEntropyLoss()
        self.train_opt = opt['train']

        # self.optimizer = Adam(self.detection_model.parameters(), lr=lr)
        ## todo: optimizer和scheduler
        # optim_params_CNN, optim_params_trans = [], []
        # name_params_CNN, name_params_trans = [], []
        # for k, v in self.detection_model.named_parameters():
        #     if v.requires_grad:
        #             name_params_trans.append(k)
        #             optim_params_trans.append(v)
           
        self.optimizer_detection = torch.optim.AdamW(self.detection_model.parameters(),
                                                                                       lr=self.opt['train']['lr_CNN'],
                                                                                       betas=(0.9, 0.999),
                                                                                       weight_decay=0.01)

        self.optimizers.append(self.optimizer_detection)
        scheduler = CosineAnnealingWarmRestarts(self.optimizer_detection, T_0=5 * len(train_loader), T_mult=1,
                                                eta_min=1e-5)
        self.schedulers.append(scheduler)

        self.optimizer_recovery = torch.optim.AdamW(self.recovery_model.parameters(),
                                                     lr=self.opt['train']['lr_CNN'],
                                                     betas=(0.9, 0.999),
                                                     weight_decay=0.01)

        self.optimizers.append(self.optimizer_recovery)
        scheduler = CosineAnnealingWarmRestarts(self.optimizer_recovery, T_0=5 * len(train_loader), T_mult=1,
                                                eta_min=1e-5)
        self.schedulers.append(scheduler)

        self.optimizer_protection = torch.optim.AdamW(self.protection_model.parameters(),
                                                    lr=self.opt['train']['lr_CNN'],
                                                    betas=(0.9, 0.999),
                                                    weight_decay=0.01)

        self.optimizers.append(self.optimizer_protection)
        scheduler = CosineAnnealingWarmRestarts(self.optimizer_protection, T_0=5 * len(train_loader), T_mult=1,
                                                eta_min=1e-5)
        self.schedulers.append(scheduler)

        ## todo: prepare inpainting model
        # self.define_inpainting_edgeconnect()
        if args.mode==self.MODE_SYNTHESIZE:
            self.define_inpainting_ZITS()
            self.define_inpainting_lama()
            self.global_step_for_inpainting = 0

    def feed_data_router(self, *, batch, mode):
        # img, img_store, label = batch
        img, label = batch
        self.data = img.cuda()
        self.label = label.cuda().float()
        self.label = self.label_cleanser(self.label)

    def feed_data_router_val(self, *, batch, mode):
        img, label = batch
        self.data = img.cuda()
        self.label = label.cuda().float()
        self.label = self.label_cleanser(self.label)

    def label_cleanser(self, label):
        for i in range(label.shape[0]):
            ratio = torch.mean(label[i])
            if ratio>0.5:
                label[i] = 1.-label[i]

        return label

    def train_ViT(self, *, epoch=None, step=None, index_info=None):
        
        return self.segment_synthesize(epoch=epoch, step=step)

    def segment_synthesize(self, *, epoch=None, step=None, index_info=None):
        # self.detection_model.eval()
        # x_hier_aug = None
        # with torch.no_grad():
        #     _, x_hier_aug = self.detection_model(self.edge)
        self.detection_model.train()
        self.protection_model.train()
        self.recovery_model.train()
        losses = 0
        with torch.enable_grad():
            if step is not None:
                self.global_step = step
            logs = {}
            lr = self.get_current_learning_rate()
            logs['lr'] = lr
    
            B = self.data.shape[0]
            ## todo: produce simulated attack, individually process the two images
            ORIGINAL_IMAGE = self.data
            # if np.random.rand()<0.5:
            #     ORIGINAL_IMAGE = self.data_augmentation_on_rendered_rgb(modified_input=ORIGINAL_IMAGE, index=np.random.randint(0,1000) % 4)
            if np.random.rand() < 0.5:
                ORIGINAL_IMAGE = self.benign_attacks_without_simulation(forward_image=ORIGINAL_IMAGE, index=np.random.randint(0,1000) % 3)
            
            TAMPER_SOURCE = torch.flip(ORIGINAL_IMAGE, dims=(0,)).clone()
    
            ## todo: protect the image
            PROTECTED_IMAGE, _ = self.protection_model(ORIGINAL_IMAGE)
            PROTECTED_IMAGE = PROTECTED_IMAGE + ORIGINAL_IMAGE
            PROTECTED_IMAGE = self.clamp_with_grad(PROTECTED_IMAGE)
            RAW_PSNR = self.psnr(self.postprocess(PROTECTED_IMAGE), self.postprocess(ORIGINAL_IMAGE)).item()
            hyper_param = 1.0*self.exponential_weight_for_backward(value=RAW_PSNR, psnr_thresh=33, exp=2)
            loss_cover = self.l1_loss(PROTECTED_IMAGE, ORIGINAL_IMAGE)
            losses += hyper_param * loss_cover
            logs['COVER'] = RAW_PSNR
            
            ## todo: tampering
            # print(f"global step {self.global_step}")
            TAMPERED_IMAGE, self.label = self.tampering_RAW(masks_GT=self.label, modified_input=PROTECTED_IMAGE,
                                                                 tamper_source=TAMPER_SOURCE,
                                                                 percent_range=(0.05, 0.25))

            ratio = torch.mean(self.label, dim=[2, 3])
            ratio = ((1 - ratio) / ratio).unsqueeze(-1).unsqueeze(-1)

            ## todo: post-process
            ATTACKED_IMAGE = TAMPERED_IMAGE
            # GT_DATA = ORIGINAL_IMAGE.clone()
            # if np.random.rand()<0.5:
            #     index_post = np.random.randint(0, 1000) % 4
            #     ATTACKED_IMAGE = self.data_augmentation_on_rendered_rgb(modified_input=ATTACKED_IMAGE, index=index_post)
            #     GT_DATA = self.data_augmentation_on_rendered_rgb(modified_input=GT_DATA, index=index_post)
            if np.random.rand() < 0.5:
                index_color = np.random.randint(0,1000) % 3
                ATTACKED_IMAGE, _, _ = self.benign_attacks(forward_image=ATTACKED_IMAGE, index=index_color)
                # GT_DATA = self.benign_attacks_without_simulation(forward_image=GT_DATA, index=index_color)


            ## todo: detect the tampering
            PREDICTED_MASK = None
            if self.opt['model_load_number'] is not None or self.global_step > 500:
                PREDICTED_MASK, _ = self.detection_model(ATTACKED_IMAGE.clone().detach().contiguous())
                sem_loss = sigmoid_focal_loss(inputs=PREDICTED_MASK, targets=self.label, alpha=0.7, gamma=1, reduction="mean")
                PREDICTED_MASK = torch.sigmoid(PREDICTED_MASK)
                f1_loss = SmoothF1(y_pred=PREDICTED_MASK, y_true=self.label)

                losses_detection = 0
                losses_detection += 1.0 * sem_loss
                # losses += 0.5 * sem_au_loss
                losses_detection += - 1.0 * f1_loss
                # losses += 1.0 * edge_loss
                logs['sem_loss'] = sem_loss
                logs['f1_loss'] = f1_loss

                losses_detection.backward()
                nn.utils.clip_grad_norm_(self.detection_model.parameters(), 1)
                if self.optimizer_detection is not None:
                    self.optimizer_detection.step()
                    self.optimizer_detection.zero_grad()

                PREDICTED_MASK, _ = self.detection_model(ATTACKED_IMAGE)
                sem_loss = sigmoid_focal_loss(inputs=PREDICTED_MASK, targets=self.label, alpha=0.7, gamma=1, reduction="mean")
                PREDICTED_MASK = torch.sigmoid(PREDICTED_MASK)
                f1_loss = SmoothF1(y_pred=PREDICTED_MASK, y_true=self.label)

                losses += 0.005 * sem_loss
                losses += - 0.005 * f1_loss
            
            ## todo: recovery of image
            RECOVERED_IMAGE = None
            BLANK_IMAGE = ATTACKED_IMAGE * (1 - self.label)
            if self.opt['model_load_number'] is not None or self.global_step>500:

                RECOVERED_IMAGE, _ = self.recovery_model(BLANK_IMAGE)
                RECOVERED_IMAGE = self.clamp_with_grad(RECOVERED_IMAGE)

                ## todo: we dont denoise the outside, since that would encourage the encoder to further prevent embedding
                NOT_STRICT_TARGET = (BLANK_IMAGE + (self.label)*ORIGINAL_IMAGE).clone().detach().contiguous()

                loss_recovery = F.l1_loss(RECOVERED_IMAGE, NOT_STRICT_TARGET, reduce=False)
                balance_map = ((1-self.label)/ratio*0.2 + self.label)
                balance_map_mean = 1 #balance_map.mean(dim=[1,2,3], keepdims=True)
                loss_recovery *= (balance_map / balance_map_mean)
                loss_recovery = loss_recovery.mean()

                RECOVERED_IMAGE = RECOVERED_IMAGE*self.label + BLANK_IMAGE

                RECOVER_PSNR = self.psnr(self.postprocess(RECOVERED_IMAGE), self.postprocess(NOT_STRICT_TARGET)).item()

                losses += loss_recovery
                logs['RECOVERY'] = RECOVER_PSNR

            else:
                RECOVERED_IMAGE, _ = self.recovery_model(BLANK_IMAGE)
                RECOVERED_IMAGE = self.clamp_with_grad(RECOVERED_IMAGE)
                RECOVER_PSNR = self.psnr(self.postprocess(RECOVERED_IMAGE), self.postprocess(BLANK_IMAGE)).item()
                loss_recovery = F.l1_loss(RECOVERED_IMAGE, BLANK_IMAGE, reduction="mean")
                losses += loss_recovery
                logs['RECOVERY'] = RECOVER_PSNR
            
            logs['loss'] = losses

            losses.backward()
            nn.utils.clip_grad_norm_(self.protection_model.parameters(), 1)
            nn.utils.clip_grad_norm_(self.recovery_model.parameters(), 1)
            if self.optimizer_protection is not None:
                self.optimizer_protection.step()
                self.optimizer_protection.zero_grad()
            if self.optimizer_recovery is not None:
                self.optimizer_recovery.step()
                self.optimizer_recovery.zero_grad()
            self.optimizer_detection.zero_grad()

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if RECOVERED_IMAGE is not None and PREDICTED_MASK is not None and \
                (self.global_step % (self.opt['restart_step'])  == (self.opt['restart_step'] - 1) or self.global_step <= 9):
            images = stitch_images(
                self.postprocess(ORIGINAL_IMAGE),
                self.postprocess(PROTECTED_IMAGE),
                self.postprocess(TAMPERED_IMAGE),
                self.postprocess(ATTACKED_IMAGE),
                self.postprocess(self.label),
                self.postprocess(PREDICTED_MASK),
                self.postprocess(RECOVERED_IMAGE),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

            print(f'Saving {name}.')

        if (self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9):
            if self.rank == 0:
                print('Saving models and training states.')
                # self.model_save(path='checkpoint/latest', epochs=self.global_step)
                self.save(accuracy=f"Epoch{epoch}", iter_label=self.global_step)

        self.global_step = self.global_step + 1

        return logs

    def exponential_weight_for_backward(self, *, value, exp=1.5, norm=-1, alpha=0.5, psnr_thresh=33, penalize=1):
        '''
            exponential loss for recovery loss's weight.
            PSNR  29     30     31     32     33(base)   34     35
            Weigh 0.161  0.192  0.231  0.277  0.333      0.400  0.500
        '''
        if 'exp_weight' in self.opt:
            exp = self.opt['exp_weight']
        if 'CE_hyper_param' in self.opt:
            alpha = self.opt['CE_hyper_param']
        if psnr_thresh is None:
            psnr_thresh = self.opt['psnr_thresh']

        item = alpha*((exp)**(norm*(value-psnr_thresh)))
        return min(max(0.25, item), 5)
    
    def test_ViT(self, epoch=None, step=None, index_info=None, running_stat=None, test_dataset_name=""):
        self.detection_model.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        with torch.no_grad():
            output, _ = self.detection_model(self.data)
            output = torch.sigmoid(output)
            # f1 = 0
            # for i in range(self.data.shape[0]):
            f1 = self.F1(output, self.label.long())
            logs[f'总AC_{test_dataset_name}'] = f1.item() #/ self.data.shape[0]
            logs[f'总AC'] = f1.item()
            logs[f'Num_总AC_{test_dataset_name}'] = 1

        if (self.global_step % 20  == 19) or self.global_step <= 9:
            images = stitch_images(
                self.postprocess(self.data),
                self.postprocess(self.label),
                self.postprocess(output),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}_{test_dataset_name}_val.png"
            images.save(name)
            print(f"Test image saved: {name}")

        if self.opt['save_checkpoint'] and index_info['is_last'] and (index_info['is_last_testset']) and \
                running_stat is not None and (running_stat['总AC'] > self.history_accuracy):
            print(f'Saving models and training states.')
            # self.model_save(path='checkpoint/latest', epochs=self.global_step)
            self.history_accuracy = running_stat['总AC']
            if 'test_save_checkpoint' in self.opt and self.opt['test_save_checkpoint']:
                self.save(accuracy=int(running_stat['总AC'] * 100), iter_label=epoch)

        return logs

    def save(self, *, accuracy, iter_label):
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/DETECT_{accuracy}_{iter_label}_ViT.pth",
            network=self.detection_model,
        )
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/PROTECT_{accuracy}_{iter_label}_ViT.pth",
            network=self.protection_model,
        )
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/RECOVERY_{accuracy}_{iter_label}_ViT.pth",
            network=self.recovery_model,
        )

