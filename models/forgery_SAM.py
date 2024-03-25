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

class forgery_SAM(BaseModel):
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
        super(forgery_SAM, self).__init__(opt, args)
        classes = [4, 4, 4, 4]
        print(f"classes: {classes}")
        if 'focal_alpha' not in self.opt:
            print("focal_alpha not in opt. using default 0.5.")
            self.opt['focal_alpha'] = 0.5
        self.focal_loss = focal_loss(alpha=self.opt['focal_alpha'], gamma=1, num_classes=4).cuda()
        self.dice_loss = DiceLoss().cuda()
        print(f"current mode is {args.mode}")
        ### todo: 定义分类网络
        if "ddpm_unet" == self.opt["network_arch"]:
            from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet
            self.detection_model = Unet(dim=64, out_channels=1, use_SRM=3, residual_connection=True, apply_bottleneck=True).cuda()
        elif "prompt_unet" == self.opt["network_arch"]:
            from Transformer_architectures.prompt_ir import PromptIR
            self.detection_model = PromptIR(decoder=True,out_channels=1).cuda()

        elif "iccv" == self.opt["network_arch"]:
            self.detection_model = my_own_elastic(nin=3, nout=1, depth=4, nch=64,
                                          num_blocks=16,
                                          use_norm_conv=False).cuda()
        elif "restormer" == self.opt["network_arch"]:
            from restoration_methods.restormer.model_restormer import Restormer
            self.detection_model = Restormer(dim=32, out_channels=1, use_SRM=3).cuda()
            # from models.vit_topdown.configs.config import get_cfg
            # cfg = get_cfg()
            # if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            #     print("prompt config loaded! ")
            #     prompt_cfg = cfg.MODEL.PROMPT
            #     print(prompt_cfg)
            # else:
            #     prompt_cfg = None
            # from models.vit_topdown.vit_top_down import vit_topdown_base_patch16_224
            # self.detection_model, self.feat_dim = vit_topdown_base_patch16_224(pretrained=False, cfg=cfg, prompt_cfg=prompt_cfg, drop_path_rate=0.1)
            # self.detection_model = self.detection_model.cuda()

            print("using restormer as testing ISP...")
        elif "topdown" == self.opt["network_arch"]:
            from restoration_methods.restormer.restormer_grouping_toast import Restormer
            self.detection_model = Restormer(img_size=self.opt['datasets']['train']['GT_size'], use_SRM=3,
                                             num_blocks=[4, 4, 4, 4], dim=32, out_channels=1).cuda()

        else:
            raise NotImplementedError("没见过的model！")

        self.detection_model = DistributedDataParallel(self.detection_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)

        ## todo: 加载之前的模型
        if self.opt['model_load_number'] is not None:
            self.reload(pretrain=f"{self.out_space_storage}/models/{self.opt['model_load_number']}",
                        network=self.detection_model,
                        strict=True)

        self.criterion = nn.CrossEntropyLoss()
        self.train_opt = opt['train']

        # self.optimizer = Adam(self.detection_model.parameters(), lr=lr)
        ## todo: optimizer和scheduler
        optim_params_CNN, optim_params_trans = [], []
        name_params_CNN, name_params_trans = [], []
        for k, v in self.detection_model.named_parameters():
            if v.requires_grad:
                # if "CNN" in k:
                #     # print(f"optim fast: {k}")
                #     name_params_CNN.append(k)
                #     optim_params_CNN.append(v)
                if "trans" in k or "position_group_tokens_" in k or \
                        "group_embeddings_" in k or \
                        "plugin_token_attention_" in k:
                    name_params_trans.append(k)
                    optim_params_trans.append(v)
                else:
                    ## todo: skip conditions
                    # if "token_attention" in opt['network_arch'] and "project_CNN" not in k:
                    #     continue
                    # if "cnn_attention" in opt['network_arch'] and \
                    #         ("project_CNN" not in k and "calibrate_CNN" not in k and "hier_MLP" not in k):
                    #     continue
                    name_params_CNN.append(k)
                    optim_params_CNN.append(v)

        self.optimizer_CNN = None if len(optim_params_CNN) == 0 else torch.optim.AdamW(optim_params_CNN,
                                                                                       lr=self.opt['train']['lr_CNN'],
                                                                                       betas=(0.9, 0.999),
                                                                                       weight_decay=0.01)
        if self.optimizer_CNN is not None:
            self.optimizers.append(self.optimizer_CNN)
            # scheduler
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
            scheduler = CosineAnnealingWarmRestarts(self.optimizer_CNN, T_0=5 * len(train_loader), T_mult=1,
                                                    eta_min=1e-5)
            self.schedulers.append(scheduler)

        self.optimizer_trans = None if len(optim_params_trans) == 0 else torch.optim.AdamW(optim_params_trans,
                                                                                           lr=self.opt['train'][
                                                                                               'lr_transformer'],
                                                                                           betas=(0.9, 0.999),
                                                                                           weight_decay=0.01)
        if self.optimizer_trans is not None:
            self.optimizers.append(self.optimizer_trans)
            scheduler = CosineAnnealingWarmRestarts(self.optimizer_trans, T_0=5 * len(train_loader), T_mult=1,
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
        if self.args.mode == self.MODE_REAL:
            img, img_store, label = batch
            self.SAM_maps = img_store.cuda()
        else:
            img, label = batch
        self.data = img.cuda()
        self.label = label.cuda().float()
        self.label = self.label_cleanser(self.label)

    def feed_data_router_val(self, *, batch, mode):
        img, img_store, label = batch
        self.data = img.cuda()
        self.label = label.cuda().float()
        self.label = self.label_cleanser(self.label)
        self.edge = img_store.cuda()

    def label_cleanser(self, label):
        for i in range(label.shape[0]):
            ratio = torch.mean(label[i])
            if ratio>0.5:
                label[i] = 1.-label[i]

        return label

    def train_ViT(self, *, epoch=None, step=None, index_info=None):
        if self.args.mode==self.MODE_REAL:
            ## todo: using CASIA2
            return self.segment_real(epoch=epoch,step=step)
        elif self.args.mode==self.MODE_SYNTHESIZE:
            ## todo: synthesize as training
            return self.segment_synthesize(epoch=epoch, step=step)

    def store_SAM_map(self, *, epoch=None, step=None, index_info=None):
        pass

    def segment_real(self, *, epoch=None, step=None, index_info=None):
        B = self.data.shape[0]
        # todo: produce simulated attack

        self.detection_model.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        with torch.enable_grad():

            prediction, x_hier = self.detection_model(self.data) #, self.SAM_maps)
            # prediction_tp, prediction_au = prediction[:B], prediction[B:]
            # prediction_dual_channel = torch.cat([1-prediction, prediction],dim=1)

            ratio = torch.mean(self.label).item()

            # sem_loss = self.bce_with_logit_loss(input=prediction_tp, target=self.label)
            sem_loss = sigmoid_focal_loss(inputs=prediction, targets=self.label, alpha=0.7, gamma=1, reduction="mean")
            # sem_au_loss = self.bce_with_logit_loss(input=prediction_au, target=torch.zeros_like(self.label).cuda())

            # prediction_tp, prediction_au = torch.sigmoid(prediction_tp), torch.sigmoid(prediction_au)
            prediction = torch.sigmoid(prediction)
            # edge_loss = sigmoid_focal_loss(inputs=prediction*self.edge, targets=self.edge, reduction="mean", alpha=0.0)
            # Lssim
            # Lssim = 1-ssim(prediction, gt) # ssim(img1,img2)
            # Liou
            f1_loss = SmoothF1(y_pred=prediction, y_true=self.label)
            # edge_loss = self.dice_loss(predict=prediction, target=self.label)
            # l1_loss = torch.mean(self.cosine_similarity(x_hier[-1], x_hier_aug[-1]))

            # if edge_loss is not None:
            losses = 0
            losses += 1.0 * sem_loss
            # losses += 0.5 * sem_au_loss
            losses += - 1.0 * f1_loss
            # losses += 1.0 * edge_loss
            # else:
            #     losses = sem_loss

            losses.backward()
            nn.utils.clip_grad_norm_(self.detection_model.parameters(), 1)
            if self.optimizer_CNN is not None:
                self.optimizer_CNN.step()
                self.optimizer_CNN.zero_grad()
            if self.optimizer_trans is not None:
                self.optimizer_trans.step()
                self.optimizer_trans.zero_grad()

            logs['loss'] = losses
            logs['sem_loss'] = sem_loss
            # logs['sem_au_loss'] = sem_au_loss
            # logs['edge_loss'] = edge_loss
            logs['f1_loss'] = f1_loss
            # logs['l1_loss'] = l1_loss

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if (self.global_step % (self.opt['restart_step']) == (self.opt['restart_step'] - 1) or self.global_step <= 9):
            images = stitch_images(
                self.postprocess(self.data),
                self.postprocess(self.label),
                self.postprocess(prediction),
                # self.postprocess(prediction_au),
                img_per_row=1
            )

            name = f"{self.out_space_storage}/images/{self.task_name}/{str(self.global_step).zfill(5)}" \
                   f"_{str(self.rank)}.png"
            images.save(name)

        if (self.global_step % (self.opt['model_save_period']) == (
                self.opt['model_save_period'] - 1) or self.global_step == 9):
            if self.rank == 0:
                print('Saving models and training states.')
                # self.model_save(path='checkpoint/latest', epochs=self.global_step)
                self.save(accuracy=f"Epoch{epoch}", iter_label=self.global_step)

        self.global_step = self.global_step + 1

        return logs

    def segment_synthesize(self, *, epoch=None, step=None, index_info=None):
        # self.detection_model.eval()
        # x_hier_aug = None
        # with torch.no_grad():
        #     _, x_hier_aug = self.detection_model(self.edge)
        B = self.data.shape[0]
        # todo: produce simulated attack, individually process the two images
        self.tamper_source = torch.flip(self.data, dims=(0,)).clone()

        if np.random.rand()<0.5:
            self.tamper_source = self.data_augmentation_on_rendered_rgb(modified_input=self.tamper_source, index=np.random.randint(0,1000) % 4)
        if np.random.rand() < 0.5:
            self.tamper_source = self.benign_attacks_without_simulation(forward_image=self.tamper_source, index=np.random.randint(0,1000) % 3)

        if np.random.rand()<0.5:
            self.data = self.data_augmentation_on_rendered_rgb(modified_input=self.data, index=np.random.randint(0,1000) % 4)
        if np.random.rand() < 0.5:
            self.data = self.benign_attacks_without_simulation(forward_image=self.data, index=np.random.randint(0,1000) % 3)


        self.simul_tampered, self.label = self.tampering_RAW(masks_GT=self.label, modified_input=self.data, tamper_source=self.tamper_source,
                                                             percent_range=(0.05,0.25))

        ## todo: combine
        self.simul_post_tampered = self.simul_tampered
        if np.random.rand()<0.5:
            self.simul_post_tampered = self.data_augmentation_on_rendered_rgb(modified_input=self.simul_post_tampered, index=np.random.randint(0,1000) % 4)
        if np.random.rand() < 0.5:
            self.simul_post_tampered = self.benign_attacks_without_simulation(forward_image=self.simul_post_tampered, index=np.random.randint(0,1000) % 3)

        self.detection_model.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        with torch.enable_grad():

            prediction, x_hier = self.detection_model(self.simul_post_tampered)
            # prediction_tp, prediction_au = prediction[:B], prediction[B:]
            # prediction_dual_channel = torch.cat([1-prediction, prediction],dim=1)

            ratio = torch.mean(self.label).item()

            # sem_loss = self.bce_with_logit_loss(input=prediction_tp, target=self.label)
            sem_loss = sigmoid_focal_loss(inputs=prediction, targets=self.label, alpha=0.7, gamma=1, reduction="mean")
            # sem_au_loss = self.bce_with_logit_loss(input=prediction_au, target=torch.zeros_like(self.label).cuda())

            # prediction_tp, prediction_au = torch.sigmoid(prediction_tp), torch.sigmoid(prediction_au)
            prediction = torch.sigmoid(prediction)
            # edge_loss = sigmoid_focal_loss(inputs=prediction*self.edge, targets=self.edge, reduction="mean", alpha=0.0)
            # Lssim
            # Lssim = 1-ssim(prediction, gt) # ssim(img1,img2)
            # Liou
            f1_loss = SmoothF1(y_pred=prediction, y_true=self.label)
            # edge_loss = self.dice_loss(predict=prediction, target=self.label)
            # l1_loss = torch.mean(self.cosine_similarity(x_hier[-1], x_hier_aug[-1]))

            # if edge_loss is not None:
            losses = 0
            losses += 1.0 * sem_loss
            # losses += 0.5 * sem_au_loss
            losses += - 1.0 * f1_loss
            # losses += 1.0 * edge_loss
            # else:
            #     losses = sem_loss

            losses.backward()
            nn.utils.clip_grad_norm_(self.detection_model.parameters(), 1)
            if self.optimizer_CNN is not None:
                self.optimizer_CNN.step()
                self.optimizer_CNN.zero_grad()
            if self.optimizer_trans is not None:
                self.optimizer_trans.step()
                self.optimizer_trans.zero_grad()

            logs['loss'] = losses
            logs['sem_loss'] = sem_loss
            # logs['sem_au_loss'] = sem_au_loss
            # logs['edge_loss'] = edge_loss
            logs['f1_loss'] = f1_loss
            # logs['l1_loss'] = l1_loss

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        if (self.global_step % (self.opt['restart_step'])  == (self.opt['restart_step'] - 1) or self.global_step <= 9):
            images = stitch_images(
                self.postprocess(self.data),
                self.postprocess(self.simul_tampered),
                self.postprocess(self.simul_post_tampered),
                self.postprocess(self.label),
                self.postprocess(prediction),
                # self.postprocess(prediction_au),
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
            pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/{accuracy}_{iter_label}_ViT.pth",
            network=self.detection_model,
        )
