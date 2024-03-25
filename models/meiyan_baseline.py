import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from models.base_model import BaseModel
from models.networks import ViT
from torch.nn.parallel import DistributedDataParallel
from losses.focal_loss import focal_loss
# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"
# from torchvision.ops.focal_loss import sigmoid_focal_loss

class meiyan_baseline(BaseModel):
    def __init__(self, opt, args, train_loader=None, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.args = args
        self.history_accuracy = 0.2
        self.methodology = opt['methodology']
        self.class_name = ["眼","提","平","白"]
        """
            prepare_networks_optimizers: set True current, preserved for future uses that only invoke static methods without creating an instances.

        """
        super(meiyan_baseline, self).__init__(opt, args)
        classes = [4,4,4,4]
        print(f"classes: {classes}")
        if 'focal_alpha' not in self.opt:
            print("focal_alpha not in opt. using default 0.5.")
            self.opt['focal_alpha'] = 0.5
        self.focal_loss = focal_loss(alpha=self.opt['focal_alpha'], gamma=1, num_classes=4).cuda()
        ### todo: 定义分类网络
        if "vit" == opt['network_arch']:
            # from transformers import ViTConfig, ViTModel
            # configuration = ViTConfig()
            # self.detection_model = ViTModel(configuration).cuda()
            from models.IFA.custom_vit import custom_viT
            self.detection_model = custom_viT(num_classes=classes, img_size=self.width_height).cuda()
        elif "VGG" == opt['network_arch']:
            print("using VGG19.")
            from CNN_architectures.pytorch_vgg_implementation import VGG_net
            self.detection_model = VGG_net(num_classes=classes).cuda()
        elif "resnet" == opt['network_arch']:
            from CNN_architectures.pytorch_resnet import ResNet50
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = ResNet50(num_classes=classes, use_SRM=True).cuda()
        elif "resnet_token_attention_plugin" == opt['network_arch']:
            from models.plug_in.resnet_plugin import resnet_plugin
            self.detection_model = resnet_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "densenet_token_attention_plugin" == opt['network_arch']:
            from models.plug_in.densenet_plugin import densenet_plugin
            self.detection_model = densenet_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "resnet_cnn_attention_plugin" == opt['network_arch']:
            from models.plug_in.resnet_cnn_plugin import resnet_cnn_plugin
            self.detection_model = resnet_cnn_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "densenet" == opt['network_arch']:
            from CNN_architectures.custom_densenet import custom_densenet
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = custom_densenet(num_classes=classes).cuda()
        elif "convnextv1" == opt['network_arch']:
            from models.IFA.convnext_official import convnext_base
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = convnext_base(num_classes=classes).cuda()
        elif "convnextv1_token_attention_plugin" == opt['network_arch']:
            from models.plug_in.convnextv1_plugin import convnextv1_plugin
            if "embed_dims" in self.opt:
                embed_dims = self.opt["embed_dims"]
            else:
                embed_dims = [96, 192, 384, 768]
            self.detection_model = convnextv1_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height, embed_dims=embed_dims).cuda()
        elif "convnextv1_cnn_attention_plugin" == opt['network_arch']:
            from models.plug_in.convnextv1_cnn_plugin import convnextv1_cnn_plugin
            self.detection_model = convnextv1_cnn_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "convnextv2" == opt['network_arch']:
            from models.IFA.convnextv2_official import convnextv2_base
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = convnextv2_base(num_classes=classes).cuda()
        elif "PVT" == opt['network_arch']:
            from models.IFA.shunted_transformer import shunted_b
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = shunted_b(img_size=self.width_height, num_classes=classes).cuda()
        elif "efficient" == opt['network_arch']:
            from CNN_architectures.pytorch_efficientnet import EfficientNet
            # self.vgg_net = ResNet50(img_channel=3, num_classes=self.unified_dim, use_SRM=True).cuda()
            # from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = EfficientNet(
                version="b5",
                num_classes=classes,
            ).cuda()
        elif "tres" == opt['network_arch']:
            from models.IFA.tres_model import Net
            self.detection_model = Net(num_classes=6).cuda()
        elif "coatnet" == opt['network_arch']:
            from models.IFA.CoAtNet import CoAtNet
            self.detection_model = CoAtNet(image_size=self.width_height, num_classes=classes).cuda()
        elif "swin" == opt['network_arch']:
            from models.IFA.Swin_official import SwinTransformer
            print(f"using swin, size {self.width_height}")
            self.detection_model = SwinTransformer(img_size=self.width_height, num_classes=classes).cuda()
        elif "conformer" == opt['network_arch']:
            from models.IFA.Conformer import Conformer
            self.detection_model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, num_classes=classes).cuda()
        elif "inception" == opt['network_arch']:
            from CNN_architectures.pytorch_inceptionet import GoogLeNet
            self.detection_model = GoogLeNet(num_classes=classes, use_SRM=False).cuda()
        elif "inception_token_attention_plugin" == opt['network_arch']:
            from models.plug_in.inception_plugin import inception_plugin
            self.detection_model = inception_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "inception_cnn_attention_plugin" == opt['network_arch']:
            from models.plug_in.inception_cnn_plugin import inception_cnn_plugin
            self.detection_model = inception_cnn_plugin(opt=self.opt, num_classes=classes, img_size=self.width_height).cuda()
        elif "cmt" == opt['network_arch']:
            from models.IFA.crossvit_official import CrossViT
            self.detection_model = CrossViT(image_size=self.width_height, num_classes=classes).cuda()
        elif "token_attention" == opt['network_arch']:
            from models.IFA.Conformer_token_attention import Conformer_hierarchical
            from functools import partial
            self.detection_model = Conformer_hierarchical(
                    img_size=self.opt['datasets']['train']['GT_size'],
                    num_classes=classes,
                    drop_tokens_per_layer=self.opt['drop_tokens_per_layer']
                ).cuda()
        elif "cnn_attention" == opt['network_arch']:
            from models.IFA.Conformer_aggregate import Conformer_hierarchical
            from functools import partial
            self.detection_model = Conformer_hierarchical(
                    img_size=self.opt['datasets']['train']['GT_size'],
                    num_classes=classes,
                    drop_tokens_per_layer=self.opt['drop_tokens_per_layer']
                ).cuda()
        elif "shunt_convnext" == opt['network_arch']:
            from models.IFA.Conformer_grouping import Conformer_hierarchical
            from functools import partial
            self.detection_model = Conformer_hierarchical(
                    img_size=self.opt['datasets']['train']['GT_size'],
                    num_classes=classes,
                    drop_tokens_per_layer=self.opt['drop_tokens_per_layer']
                ).cuda()
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

        self.optimizer_CNN = None if len(optim_params_CNN)==0 else torch.optim.AdamW(optim_params_CNN,
                                      lr=self.opt['train']['lr_CNN'], betas=(0.9, 0.999), weight_decay=0.01)
        if self.optimizer_CNN is not None:
            self.optimizers.append(self.optimizer_CNN)
            # scheduler
            # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
            scheduler = CosineAnnealingWarmRestarts(self.optimizer_CNN, T_0=5*len(train_loader),T_mult=1,eta_min=1e-5)
            self.schedulers.append(scheduler)

        self.optimizer_trans = None if len(optim_params_trans) == 0 else torch.optim.AdamW(optim_params_trans,
                                      lr=self.opt['train']['lr_transformer'],betas=(0.9, 0.999),weight_decay=0.01)
        if self.optimizer_trans is not None:
            self.optimizers.append(self.optimizer_trans)
            scheduler = CosineAnnealingWarmRestarts(self.optimizer_trans, T_0=5*len(train_loader),T_mult=1,eta_min=1e-5)
            self.schedulers.append(scheduler)

    def feed_data_router(self, *, batch, mode):
        data, label, magnitude, img_path = batch
        self.data = data.cuda()
        if isinstance(label, list):
            label = torch.stack(label, dim=1)
            magnitude = torch.stack(magnitude, dim=1)
        self.label = label.long().cuda()
        self.magnitude = magnitude.long().cuda()

    def feed_data_router_val(self, *, batch, mode):
        data, label, magnitude, img_path = batch
        self.data = data.cuda()
        if isinstance(label, list):
            label = torch.stack(label, dim=1)
            magnitude = torch.stack(magnitude, dim=1)
        self.label = label.long().cuda()
        self.magnitude = magnitude.long().cuda()

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
            if index_info['begin_10']:
                for i in range(self.magnitude.shape[0]):
                    print(self.magnitude[i])
            ## outputs a list, [[大眼的token，美白的token，...]]
            output = self.detection_model(self.data)
            acc, abs_correct = None, None
            loss, loss_list = 0, []
            ## todo: note the format: way1(conv)-[cls1, cls2, cls3], way2(trans)-[cls1, cls2, cls3]
            for i in range(self.magnitude.shape[1]):

                loss_class = [self.focal_loss(o[i], self.magnitude[:,i]) / len(output) for o in output]
                loss_class = sum(loss_class)
                loss_CE = [self.criterion(o[i], self.magnitude[:, i]) / len(output) for o in output]
                loss_CE = sum(loss_CE)
                # loss += loss_class
                loss_list.append(loss_class)
                # logs[f'{self.class_name[i]}'] = loss_class.item()
                logs[f'{self.class_name[i]}_CE'] = loss_CE.item()

                ## todo: relative
                pred_binary = (sum([o[i] for o in output]).argmax(dim=1) > 0).float()
                gt_binary = (self.magnitude[:, i] > 0).float()
                new_acc = pred_binary.isclose(gt_binary).float()
                ## todo: absolute
                new_abs_correct = (sum([o[i] for o in output]).argmax(dim=1) == self.magnitude[:, i]).float()

                logs[f'均{self.class_name[i]}'] = new_acc.mean().item()
                logs[f'全{self.class_name[i]}'] = new_abs_correct.mean().item()
                # logs[f'l{self.class_name[i]}'] = loss_class.item()
                logs[f'CE{self.class_name[i]}'] = loss_CE.item()
                acc = new_acc if acc is None else acc * new_acc
                abs_correct = new_abs_correct if abs_correct is None else abs_correct * new_abs_correct
            logs['总均'] = acc.mean().item()
            logs['总AC'] = abs_correct.mean().item()
            ## todo: grad norm
            λ = None #self.calculate_lambda(losses=loss_list,last_layer=self.detection_model.module.get_last_layer(), logs=logs)
            for i in range(len(loss_list)):
                loss += (1 if λ is None else λ[i]) * loss_list[i]
                # logs[f'λ{self.class_name[i]}'] = λ[i]
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

        if (self.global_step % (self.opt['model_save_period']) == (self.opt['model_save_period'] - 1) or self.global_step == 9):
            if self.rank == 0:
                print('Saving models and training states.')
                # self.model_save(path='checkpoint/latest', epochs=self.global_step)
                self.save(accuracy=f"Epoch{epoch}", iter_label=self.global_step)

        self.global_step = self.global_step + 1

        return logs

    def calculate_lambda(self, *, losses, last_layer, logs):
        ## todo: original version
        # last_layer = self.decoder.model[-1]
        # last_layer_weight = last_layer.weight
        # nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        # g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]
        #
        # λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # λ = torch.clamp(λ, 0, 1e4).detach()
        # return 0.8 * λ

        last_layer_weight = last_layer.weight
        minimux_grad, avg_grad, grads = 1e6, 0, []
        for loss in losses:
            cls_grads = torch.norm(torch.autograd.grad(loss, last_layer_weight, retain_graph=True)[0])
            grads.append(cls_grads)
            minimux_grad = min(minimux_grad,cls_grads)
            avg_grad += cls_grads

        avg_grad /= len(losses)
        λ = [torch.clamp(avg_grad/grad, 0, 1e4).detach() for grad in grads]
        return λ

    def test_ViT(self, epoch=None, step=None, index_info=None, running_stat=None):
        self.detection_model.eval()
        if step is not None:
            self.global_step = step
        logs = {}
        lr = self.get_current_learning_rate()
        logs['lr'] = lr
        pred_matrix = None
        with torch.no_grad():
            output = self.detection_model(self.data)
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

    # def model_save(self, *, path, epochs):
    #     checkpoint = {'models': self.detection_model,
    #                   'model_state_dict': self.detection_model.state_dict(),
    #                   'optimizer_state_dict': self.optimizer.state_dict(),
    #                   'epochs': epochs,
    #                   # 'epoch_acc_list': epoch_acc_list,
    #                   # 'epoch_loss_list': epoch_loss_list}
    #                   }
    #     torch.save(checkpoint, f'{path}.pkl')

    # def model_load(self):
    #     from util import load_checkpoint
    #     self.detection_model, result = load_checkpoint('checkpoint.pkl')
    #     acc = result['epoch_acc_list']
    #     plt.plot(acc)
    #     loss_list = result['epoch_loss_list']
    #     plt.plot(loss_list)

        # # todo: 预测k张图片
        # k = 3
        # sample_idx = np.random.randint(1, len(test_list), size=k)
        # sample_test_list = [test_list[i] for i in sample_idx]
        # sample_test_labels = [test_labels[i] for i in sample_idx]
        # sample_test_data = meiyanDataset(sample_test_list, sample_test_labels, transform=test_transforms)
        # sample_test_loader = DataLoader(dataset=sample_test_data, batch_size=1, shuffle=True)
        #
        # models.eval()
        # i = 0
        # for data, label in tqdm(sample_test_loader):
        #     data = data.cuda()
        #     label = label.cuda()
        #
        #     output = models(data)  # 此时输出是11维，选择概率最大的
        #     pred = output.argmax(dim=1).cpu().numpy()
        #     img = Image.open(test_list[sample_idx[i]])
        #     i += 1
        #     plt.subplot(1, k, i)
        #     title = "label:" + str(label.cpu().numpy()[0]) + ",pred:" + str(pred[0])
        #     plt.title(title)
        #     plt.imshow(img)



