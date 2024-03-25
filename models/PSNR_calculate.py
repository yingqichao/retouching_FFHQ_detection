import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from models.base_model import BaseModel
from models.networks import ViT
from torch.nn.parallel import DistributedDataParallel
from losses.focal_loss import focal_loss
import os
import cv2
import torchvision.transforms as transforms
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
# from torchvision.ops.focal_loss import sigmoid_focal_loss
from utils import stitch_images

class PSNR_calculate(BaseModel):
    def __init__(self, opt, args, train_loader=None, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.args = args
        self.history = 0
        self.count = 0
        self.methodology = opt['methodology']
        self.class_name = ["眼", "提", "平", "白"]
        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(PSNR_calculate, self).__init__(opt, args)
        classes = [4, 4, 4, 4]
        print(f"classes: {classes}")
        if 'focal_alpha' not in self.opt:
            print("focal_alpha not in opt. using default 0.5.")
            self.opt['focal_alpha'] = 0.5
        self.focal_loss = focal_loss(alpha=self.opt['focal_alpha'], gamma=1, num_classes=4).cuda()
        ### todo: 定义分类网络
        self.detection_model_copy = None
        if "convunet" == opt['network_arch']:
            from models.plug_in.convunet import ConvUnet_plugin
            self.detection_model = ConvUnet_plugin(opt=self.opt, num_classes=classes,
                                                    img_size=self.width_height).cuda()
            self.detection_model_copy = ConvUnet_plugin(opt=self.opt, num_classes=classes,
                                                        img_size=self.width_height).cuda()
        elif "ddpm_unet" == opt['network_arch']:
            from models.plug_in.ddpm_unet_plugin import ddpm_unet_plugin
            self.detection_model = ddpm_unet_plugin(opt=self.opt, num_classes=classes,img_size=self.width_height,
                                                    dim=32, embed_dims=[64,128,256,512],).cuda()
            self.detection_model_copy = ddpm_unet_plugin(opt=self.opt, num_classes=classes,img_size=self.width_height,
                                                    dim=32, embed_dims=[64,128,256,512],).cuda()
        else:
            raise NotImplementedError("模型结构不对，请检查！")
        print(f"{opt['network_arch']} models created. method: {opt['methodology']}")
        self.detection_model = DistributedDataParallel(self.detection_model,
                                                       device_ids=[torch.cuda.current_device()],
                                                       find_unused_parameters=True)


        ## todo: 加载之前的模型
        if self.opt['model_load_number'] is not None:
            self.reload(pretrain=f"{self.out_space_storage}/models/{self.opt['model_load_number']}",
                        network=self.detection_model,
                        strict=False)

        ## todo: copy identical model for recovery
        if self.detection_model_copy is not None:
            self._momentum_update_key_encoder()
            for param_copy in self.detection_model_copy.parameters():
                param_copy.requires_grad = False

            self.detection_model_copy.eval()

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

        self.history_accuracy = 0.0

    def feed_data_router(self, *, batch, mode):
        data, data_origin, label, magnitude, img_path, img_ori_path = batch
        self.data = data.cuda()
        self.data_origin = data_origin.cuda()
        if isinstance(label, list):
            label = torch.stack(label, dim=1)
            magnitude = torch.stack(magnitude, dim=1)
        self.label = label.long().cuda()
        self.magnitude = magnitude.long().cuda()

    def feed_data_router_val(self, *, batch, mode):
        data, data_origin, label, magnitude, img_path, img_ori_path = batch
        self.data = data.cuda()
        self.data_origin = data_origin.cuda()
        if isinstance(label, list):
            label = torch.stack(label, dim=1)
            magnitude = torch.stack(magnitude, dim=1)
        self.label = label.long().cuda()
        self.magnitude = magnitude.long().cuda()

    # def get_gradcam(self, *, batch, filename):
    #     data, label, magnitude, img_path = batch
    #     self.data = data.cuda()
    #     if isinstance(label, list):
    #         label = torch.stack(label, dim=1)
    #         magnitude = torch.stack(magnitude, dim=1)
    #     self.label = label.long().cuda()
    #     self.magnitude = magnitude.long().cuda()
    #
    #
    #     # 存放梯度和特征图
    #     self.fmap_block = list()
    #     self.grad_block = list()
    #
    #     # 图片读取；网络加载
    #     net = self.detection_model()
    #     net.eval()  # 一定要加上net.eval()，不然深一点的网络(如resnet)就会识别出错，而且每次执行后的类激活图都不一样
    #     # net.load_state_dict(torch.load(path_net)) # 读入已学习的网络
    #
    #     # 注册hook,位于最后一个卷积层
    #     net.conv2.register_forward_hook(self.farward_hook)
    #     net.conv2.register_backward_hook(self.backward_hook)
    #
    #     # forward
    #     self.output = net(self.data)
    #
    #     # classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #     # idx = np.argmax(self.output.cpu().data.numpy())
    #     # print("predict: {}".format(classes[idx]))
    #
    #     # backward
    #     net.zero_grad()
    #     class_loss = self.comp_class_vec(self.output)  # 此处不是损失函数的概念，而是激活值的概念
    #     class_loss.backward()
    #
    #     # 生成cam
    #     grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
    #     fmap = self.fmap_block[0].cpu().data.numpy().squeeze()
    #
    #     cam = self.gen_cam(fmap, grads_val)
    #
    #     # 保存cam图片
    #     for i in range(self.data.shape[0]):
    #         data = self.data[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #         img_show = np.float32(cv2.resize(data, (224, 224))) / 255
    #         self.show_cam_on_image(img_show, cam,
    #                                f"{self.out_space_storage}/models/{self.opt['task_name']}/{filename[i]}_CAM.png"
    #                                )
    #

    def train_ViT(self,*, epoch=None, step=None, index_info=None):
        self.detection_model.train()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr

        ## todo: due to tencent does not
        if self.opt["train_dataset"] == "tencent":
            self.magnitude[:, 2] = (self.magnitude[:, 2] > 0).long()

        with torch.enable_grad():
            ## todo: CLASSIFICATION STAGE
            if "convunet" == self.opt['network_arch']:
                if self.opt["train_recover"]:
                    pred_origin, output = self.detection_model(self.data)
                    pred_origin = self.clamp_with_grad(pred_origin)
                else:
                    output = self.detection_model.module.forward_classification(self.data)
            elif "ddpm_unet" == self.opt['network_arch']:
                output = self.detection_model.module.forward_classification(self.data, time=None)

            acc, abs_correct = None, None
            loss, loss_list = 0, []
            ###### note the format: way1(conv)-[cls1, cls2, cls3], way2(trans)-[cls1, cls2, cls3]
            for i in range(self.magnitude.shape[1]):

                # loss_class = [self.focal_loss(o[i], self.magnitude[:, i]) / len(output) for o in output]
                # loss_class = sum(loss_class)
                loss_CE = [self.criterion(o[i], self.magnitude[:, i]) / len(output) for o in output]
                loss_CE = sum(loss_CE)
                # loss += loss_class
                loss_list.append(loss_CE)
                # logs[f'{self.class_name[i]}'] = loss_class.item()
                logs[f'{self.class_name[i]}_CE'] = loss_CE.item()

                ######## relative
                pred_binary = (sum([o[i] for o in output]).argmax(dim=1) > 0).float()
                gt_binary = (self.magnitude[:, i] > 0).float()
                new_acc = pred_binary.isclose(gt_binary).float()
                ######## absolute
                pred_largest = sum([o[i] for o in output]).argmax(dim=1)
                if index_info['begin_10']:
                    for _ in range(self.magnitude.shape[0]):
                        # print("label:" + self.magnitude[i])
                        print(pred_largest)

                new_abs_correct = (pred_largest == self.magnitude[:, i]).float()

                logs[f'均{self.class_name[i]}'] = new_acc.mean().item()
                logs[f'全{self.class_name[i]}'] = new_abs_correct.mean().item()
                # logs[f'l{self.class_name[i]}'] = loss_class.item()
                logs[f'CE{self.class_name[i]}'] = loss_CE.item()
                acc = new_acc if acc is None else acc * new_acc
                abs_correct = new_abs_correct if abs_correct is None else abs_correct * new_abs_correct
            logs['总均'] = acc.mean().item()
            logs['总AC'] = abs_correct.mean().item()
            ######## balance multi-task (not used)
            λ = None  # self.calculate_lambda(losses=loss_list,last_layer=self.detection_model.module.get_last_layer(), logs=logs)
            for i in range(len(loss_list)):
                loss += (1 if λ is None else λ[i]) * loss_list[i]
                # logs[f'λ{self.class_name[i]}'] = λ[i]

            ## todo: RECOVERY STAGE
            if self.opt["train_recover"]:
                if "convunet" == self.opt['network_arch']:
                    pass
                elif "ddpm_unet" == self.opt['network_arch']:
                    pred_origin, _ = self.detection_model(self.data, time=self.magnitude)
                    pred_origin = self.clamp_with_grad(pred_origin)

                l_recover = self.l1_loss(pred_origin, self.data_origin)
                l_rec_cls = 0

                if "convunet" == self.opt['network_arch']:
                    output_pred_recovery = self.detection_model_copy.forward_classification(pred_origin)
                elif "ddpm_unet" == self.opt['network_arch']:
                    output_pred_recovery = self.detection_model_copy.forward_classification(pred_origin, time=None)

                for i in range(self.magnitude.shape[1]):
                    # l_rec_subcls = [self.focal_loss(o[i], self.magnitude[:, i].zero_()) / len(output) for o in output.detach()]
                    l_rec_subcls = [self.criterion(o[i], self.magnitude[:, i]) / len(output) for o in output_pred_recovery]
                    l_rec_cls += sum(l_rec_subcls)
                PSNR_recover = self.psnr(self.postprocess(pred_origin),
                                   self.postprocess(self.data_origin)).item()
                logs['l_rec_PSNR'] = PSNR_recover
                logs['l_rec_cls'] = l_rec_cls.item()
                loss += l_rec_cls/4
                loss += 2*l_recover

            loss.backward()
            nn.utils.clip_grad_norm_(self.detection_model.parameters(), 1)
            if self.optimizer_CNN is not None:
                self.optimizer_CNN.step()
                self.optimizer_CNN.zero_grad()
            if self.optimizer_trans is not None:
                self.optimizer_trans.step()
                self.optimizer_trans.zero_grad()

            # emsemble_prediction = 0
            # for idx, item in enumerate(output):
            #     name_acc = ['CNN', 'trans']
            #     emsemble_prediction += item[0]
            #     acc = (item[0].argmax(dim=1) == self.label).float().mean()
            #     logs[f'{name_acc[idx]}_acc'] = acc
            # acc = (emsemble_prediction.argmax(dim=1) == self.label).float().mean()
            # logs['overall_acc'] = acc
            logs['loss'] = loss

        ######## Finally ####################
        for scheduler in self.schedulers:
            scheduler.step()

        ## todo: momemtum update
        if self.detection_model_copy is not None and self.opt["train_recover"]:
            self._momentum_update_key_encoder()

        if (self.global_step % (self.opt['restart_step'])  == (
                self.opt['restart_step'] - 1) or self.global_step == 9) and self.opt["train_recover"]:
            images = stitch_images(
                self.postprocess(self.data),
                self.postprocess(self.data_origin),
                self.postprocess(5 * torch.abs(self.data_origin - self.data)),
                self.postprocess(pred_origin),
                self.postprocess(5 * torch.abs(self.data_origin - pred_origin)),
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

    ####################################################################################################
    # todo:  Momentum update of the key encoder
    # todo: param_k: momentum
    ####################################################################################################
    @torch.no_grad()
    def _momentum_update_key_encoder(self, momentum=0.9):
        for param_q, param_k in zip(self.detection_model.parameters(), self.detection_model_copy.parameters()):
            param_k.data = param_q.data


    def test_ViT(self, epoch=None, step=None, index_info=None, running_stat=None):
        self.detection_model.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        pred_matrix = None
        with torch.no_grad():
            pred_origin, output = self.detection_model(self.data)

            ## todo: evaluate image recovery
            PSNR_recover = self.psnr(self.postprocess(pred_origin),
                                               self.postprocess(self.data_origin)).item()
            logs['l_rec_PSNR'] = PSNR_recover

            loss = 0
            sum_tp, sum_tn, sum_abs_correct = None, None, None
            ## todo: note the format: way1(conv)-[cls1, cls2, cls3], way2(trans)-[cls1, cls2, cls3]
            for i in range(self.magnitude.shape[1]):
                if index_info['begin_10']:
                    print(output[0][i])
                    print(self.magnitude[:,i])

                ## todo: due to tencent does not
                if self.opt["test_dataset"] == "tencent" and self.class_name[i] == "平":
                    self.magnitude[:, i] = (self.magnitude[:, i] > 0).long()

                # loss_class = [self.focal_loss(o[i], self.magnitude[:, i]) / len(output) for o in output]
                # loss_class = sum(loss_class)
                loss_CE = [self.criterion(o[i], self.magnitude[:, i]) / len(output) for o in output]
                loss_CE = sum(loss_CE)
                loss += loss_CE

                ## todo: relative binarized results
                pred = sum([o[i] for o in output])[:,:2].argmax(dim=1)
                # gt_binary = (self.magnitude[:, i] > 0).float()

                pred_matrix = torch.cat([pred_matrix,pred.unsqueeze(1)],dim=1) if pred_matrix is not None else pred.unsqueeze(1)

                ## todo: deprecated due to 容易出错
                # tp_mask = gt_binary
                # tn_mask = 1-gt_binary
                # ## todo: relative - TP / TN
                # pred_binary_tp = pred_binary*tp_mask
                # gt_binary_tp = gt_binary*tp_mask
                # pred_binary_tn = pred_binary*tn_mask
                # gt_binary_tn = gt_binary*tn_mask
                #
                # new_acc_tp = pred_binary_tp.isclose(gt_binary_tp).float()
                # new_acc_tn = pred_binary_tn.isclose(gt_binary_tn).float()
                #
                # ## todo: Absolute Correctness
                # new_abs_correct = (sum([o[i] for o in output]).argmax(dim=1)==self.magnitude[:,i]).float()
                #
                # TP_single_type = (new_acc_tp * gt_binary_tp).sum() / (gt_binary_tp.sum())
                # TN_single_type = (new_acc_tn * gt_binary_tn).sum() / (gt_binary_tn.sum())
                # logs[f'均{self.class_name[i]}'] = TP_single_type.item()
                # logs[f'TN{self.class_name[i]}'] = TN_single_type.item()
                # logs[f'全{self.class_name[i]}'] = new_abs_correct.mean().item()
                # # logs[f'l{self.class_name[i]}'] = loss_class.item()
                # logs[f'CE{self.class_name[i]}'] = loss_CE.item()
                # sum_tp = new_acc_tp if sum_tp is None else sum_tp * new_acc_tp
                # sum_tn = new_acc_tn if sum_tn is None else sum_tn * new_acc_tn
                # sum_abs_correct = new_abs_correct if sum_abs_correct is None else sum_abs_correct*new_abs_correct

            gt_binary = (self.magnitude > 0).long()
            pred_binary_matrix = (pred_matrix>0).long()

            ## todo: evaluation protocol on single type
            for j_type in range(len(self.class_name)):
                TP_correct, TP_count, TN_correct, TN_count, AC = 0, 0, 0, 0, 0
                for i_batch in range(self.magnitude.shape[0]):
                    if (self.magnitude[i_batch, j_type]==pred_matrix[i_batch, j_type]):
                        AC += 1

                    if gt_binary[i_batch,j_type]==0:
                        TN_count += 1
                        if gt_binary[i_batch,j_type]==pred_binary_matrix[i_batch,j_type]:
                            TN_correct += 1
                    else:
                        TP_count += 1
                        if gt_binary[i_batch, j_type]==pred_binary_matrix[i_batch, j_type]:
                            TP_correct += 1

                logs[f'TP{self.class_name[j_type]}'] = TP_correct
                logs[f'Num_TP{self.class_name[j_type]}'] = TP_count
                logs[f'TN{self.class_name[j_type]}'] = TN_correct
                logs[f'Num_TN{self.class_name[j_type]}'] = TN_count
                logs[f'AC{self.class_name[j_type]}'] = AC/self.magnitude.shape[0]

            ## todo: evaluation protocol by accumulation
            TPsum_correct, TPsum_count, TNsum_correct, TNsum_count, AC, AC_count = 0, 0, 0, 0, 0, 0
            for i_batch in range(self.magnitude.shape[0]):
                TP_correct, TP_count, TN_correct, TN_count, AC = 0, 0, 0, 0, 0
                for j_type in range(len(self.class_name)):
                    if (self.magnitude[i_batch, j_type]==pred_matrix[i_batch, j_type]):
                        AC += 1

                    if gt_binary[i_batch, j_type] == 0:
                        TN_count += 1
                        if gt_binary[i_batch, j_type]==pred_binary_matrix[i_batch, j_type]:
                            TN_correct += 1
                    else:
                        TP_count += 1
                        if gt_binary[i_batch, j_type]==pred_binary_matrix[i_batch, j_type]:
                            TP_correct += 1

                if AC==4:
                    AC_count += 1
                if TP_count!=0:
                    TPsum_count += 1
                    if TP_correct==TP_count:
                        TPsum_correct += 1
                if TN_count != 0:
                    TNsum_count += 1
                    if TN_correct == TN_count:
                        TNsum_correct += 1

            logs['总TP'] = TPsum_correct
            logs['Num_总TP'] = TPsum_count
            logs['总TN'] = TNsum_correct
            logs['Num_总TN'] = TNsum_count
            logs['总AC'] = AC_count / self.magnitude.shape[0]

                # # emsemble_prediction = 0
                # for idx, item in enumerate(output):
                #     # emsemble_prediction += item[0]
                #     acc = (item[idx].argmax(dim=1)>0 == self.magnitude[idx]>0).float().mean()
                #     sub_accuracy[idx] += acc.item() / len(self.val_loader)
                # # acc = ((emsemble_prediction).argmax(dim=1) == self.label).float().mean()
                #
                # epoch_accuracy += acc.item() / len(self.val_loader)
                # epoch_loss += loss.item() / len(self.val_loader)
                # # if idx%10==9 or idx==len(self.val_loader)-1:
                # print(f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - history: {self.history_accuracy:.4f} "
                #       f"- sub {sub_accuracy}\n")

        if self.opt['save_checkpoint'] and index_info['is_last'] and running_stat is not None and running_stat['总TP'] > self.history_accuracy:
            print(f'Saving models and training states.')
            # self.model_save(path='checkpoint/latest', epochs=self.global_step)
            self.history_accuracy = running_stat['总TP']
            if 'test_save_checkpoint' in self.opt and self.opt['test_save_checkpoint']:
                self.save(accuracy=int(running_stat['总TP'] * 100), iter_label=epoch)


        return logs

    def save(self, *, accuracy, iter_label):
        self.save_network(
            pretrain=f"{self.out_space_storage}/models/{self.opt['task_name']}/{accuracy}_{iter_label}_ViT.pth",
            network=self.detection_model,
        )

    def img_transform(self, img_in, transform):
        """
        将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
        :param img_roi: np.array
        :return:
        """
        img = img_in.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img)
        img = img.unsqueeze(0)  # C*H*W --> B*C*H*W,增加第一维的batch通道，使得图片能够输入网络
        return img

    def img_preprocess(self,img_in):
        """
        读取图片，转为模型可读的形式
        :param img_in: ndarray, [H, W, C]
        :return: PIL.image
        """
        img = img_in.copy()
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1]  # BGR --> RGB,之后读取图片会使用opencv读取，读取的颜色通道为BGR，为了适应模型，需要将颜色通道转回为RGB。
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
        ])
        img_input = self.img_transform(img, transform)
        return img_input

    # 定义获取梯度的函数
    def backward_hook(self,module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self,module, input, output):
        self.fmap_block.append(output)

    def show_cam_on_image(self,img, mask, out_dir):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        path_cam_img = os.path.join(out_dir, "cam.jpg")
        path_raw_img = os.path.join(out_dir, "raw.jpg")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(path_cam_img, np.uint8(255 * cam))
        cv2.imwrite(path_raw_img, np.uint8(255 * img))

    def comp_class_vec(self,ouput_vec, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(ouput_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot * self.output)  # one_hot = 11.8605
        return class_vec

    def gen_cam(self,feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:],
                       dtype=np.float32)  # cam shape (H, W),feature_map.shape[1:] 表示取第一维度及之后的其余维度的尺寸，如 [512, 14, 14] --> (14, 14)

        weights = np.mean(grads, axis=(1, 2))  # 计算每个通道的权重均值

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]  # 将梯度权重与特征图相乘再累加

        cam = np.maximum(cam, 0)  # relu
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)  # 归一化

        return cam

