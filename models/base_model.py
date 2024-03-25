import copy
# from .networks import SPADE_UNet
import os
from collections import OrderedDict

import numpy as np
import torch
from skimage.feature import canny
import cv2
import torch.distributed as dist
import torch.nn as nn
# from data.pipeline import pipeline_tensor2image
# import matlab.engine
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as Functional
from PIL import Image
from skimage.color import rgb2gray
from torch.nn.parallel import DistributedDataParallel

from noise_layers import *
from noise_layers.dropout import Dropout
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.resize import Resize
from utils.JPEG import DiffJPEG
from utils.commons import create_folder
# from MVSS.models.mvssnet import get_mvss
# from MVSS.models.resfcn import ResFCN
from utils.metrics import PSNR
from omegaconf import OmegaConf
import yaml

class BaseModel():
    def __init__(self, opt,  args):
        ### todo: options
        self.opt = opt
        # self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

        self.rank = torch.distributed.get_rank()
        self.opt = opt
        self.args = args
        # self.train_opt = opt['train']
        # self.test_opt = opt['test']


        ### todo: constants
        self.global_step = 0
        self.global_step_for_inpainting = 0
        self.width_height = opt['datasets']['train']['GT_size']

        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

        ### todo: network definitions
        self.network_list = []
        self.real_H, self.real_H_path, self.previous_images = None, None, None
        self.previous_previous_images, self.previous_previous_canny = None, None
        self.previous_protected, self.previous_canny = None, None


        self.out_space_storage = "/groupshare/meiyan_detection_results"
        self.task_name = opt['task_name']
        self.create_folders_for_the_experiment()

        ### todo: losses and attack layers
        self.psup = nn.PixelShuffle(upscale_factor=2).cuda()
        self.psdown = nn.PixelUnshuffle(downscale_factor=2).cuda()
        self.tanh = nn.Tanh().cuda()
        self.psnr = PSNR(255.0).cuda()
        # self.lpips_vgg = lpips.LPIPS(net="vgg").cuda()
        # self.exclusion_loss = ExclusionLoss().type(torch.cuda.FloatTensor).cuda()

        self.crop = Crop().cuda()
        self.dropout = Dropout().cuda()
        self.gaussian = Gaussian().cuda()
        self.salt_pepper = SaltPepper(prob=0.01).cuda()
        self.gaussian_blur = GaussianBlur(opt=self.opt).cuda()
        self.median_blur = MiddleBlur(opt=self.opt).cuda()
        self.resize = Resize(opt=self.opt).cuda()
        self.identity = Identity().cuda()

        self.jpeg_simulate = [
            [DiffJPEG(50, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(55, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(60, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(65, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(70, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(75, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(80, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(85, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(90, height=self.width_height, width=self.width_height).cuda(), ]
            , [DiffJPEG(95, height=self.width_height, width=self.width_height).cuda(), ]
        ]

        self.bce_loss = nn.BCELoss().cuda()
        self.bce_with_logit_loss = nn.BCEWithLogitsLoss().cuda()
        self.l1_loss = nn.SmoothL1Loss(beta=0.5).cuda()  # reduction="sum"
        self.cosine_similarity = nn.CosineSimilarity().cuda()
        self.l2_loss = nn.MSELoss().cuda()  # reduction="sum"
        self.CE_loss = nn.CrossEntropyLoss().cuda()



    ### todo: Abstract Methods
    def define_ddpm_unet_network(self, dim = 32, **kwargs):
        from CNN_architectures.ddpm_lucidrains import Unet
        # input = torch.ones((3, 3, 128, 128)).cuda()
        # output = model(input, torch.zeros((1)).cuda())

        print(f"using ddpm_unet, {kwargs}")
        model = Unet(dim=dim, **kwargs).cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        return model


    def print_this_image(self, image, filename):
        '''
            the input should be sized [C,H,W], not [N,C,H,W]
        '''
        camera_ready = image.unsqueeze(0)
        torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                     filename, nrow=1,
                                     padding=0, normalize=False)

    ### todo: optimizer
    def create_optimizer(self, net, lr=1e-4, weight_decay=0):
        ## lr should be train_opt['lr_scratch'] in default
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            # else:
            #     if self.rank <= 0:
            #         print('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.AdamW(optim_params, lr=lr,
                                      weight_decay=weight_decay,
                                      betas=(0.9, 0.99))  # train_opt['beta1'], train_opt['beta2']
        self.optimizers.append(optimizer)

        return optimizer

    ### todo: folders
    def create_folders_for_the_experiment(self):
        create_folder(self.out_space_storage)
        create_folder(self.out_space_storage + "/models")
        create_folder(self.out_space_storage + "/images")
        create_folder(self.out_space_storage + "/isp_images/")
        create_folder(self.out_space_storage + "/models/" + self.task_name)
        create_folder(self.out_space_storage + "/images/" + self.task_name)
        create_folder(self.out_space_storage + "/isp_images/" + self.task_name)

    def load_model_wrapper(self,*,folder_name,model_name,network, network_name, strict=True):
        load_detector_storage = self.opt[folder_name]
        model_path = str(self.opt[model_name])  # last time: 10999
        load_models = self.opt[model_name] > 0
        if load_models:
            print(f"loading models: {network_name}")
            pretrain = load_detector_storage + model_path
            self.reload(pretrain, network, strict=strict)

    def reload(self, pretrain, network, strict=True):
        load_path_G = pretrain
        if load_path_G is not None:
            print('Loading models for class [{:s}] ...'.format(load_path_G))
            if os.path.exists(load_path_G):
                self.load_network(load_path_G, network, strict=strict)
            else:
                print('Did not find models for class [{:s}] ...'.format(load_path_G))

    ### todo: trivial stuffs
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def random_float(self, min, max):
        """
        Return a random number
        :param min:
        :param max:
        :return:
        """
        return np.random.rand() * (max - min) + min

    def get_paths_from_images(self, path):
        '''
            get image path list from image folder
        '''
        # assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
        if path is None:
            return None, None

        images_dict = {}
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if self.is_image_file(fname):
                    img_path = os.path.join(dirpath, fname)
                    # images.append((path, dirpath[len(path) + 1:], fname))
                    images_dict[fname] = img_path
        assert images_dict, '{:s} has no valid image file'.format(path)

        return images_dict

    def print_individual_image(self, cropped_GT, name):
        for image_no in range(cropped_GT.shape[0]):
            camera_ready = cropped_GT[image_no].unsqueeze(0)
            torchvision.utils.save_image((camera_ready * 255).round() / 255,
                                         name, nrow=1, padding=0, normalize=False)


    def load_image(self, path, readimg=False, grayscale=False, require_canny=False):
        import data.util as util
        GT_path = path

        img_GT = util.read_img(GT_path)

        # change color space if necessary
        img_GT = util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]
        if grayscale:
            img_GT = rgb2gray(img_GT)

        img_GT = cv2.resize(copy.deepcopy(img_GT), (self.width_height, self.width_height),
                            interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not grayscale:
            image = img_GT[:, :, [2, 1, 0]]
            image = torch.from_numpy(
                np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
        else:
            image = torch.from_numpy(
                np.ascontiguousarray(img_GT)).float()

        if require_canny and not grayscale:
            img_gray = rgb2gray(img_GT)
            sigma = 2  # random.randint(1, 4)
            cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float)
            canny_image = torch.from_numpy(
                np.ascontiguousarray(cannied)).float()
            return image.cuda().unsqueeze(0), canny_image.cuda().unsqueeze(0).unsqueeze(0)
        else:
            return image.cuda().unsqueeze(0)

    def tensor_to_image(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def tensor_to_image_batch(self, tensor):

        tensor = tensor * 255.0
        image = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
        # image = tensor.permute(0,2,3,1).detach().cpu().numpy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def image_to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(np.asarray(img)).float()
        return img_t

    def clamp_with_grad(self, tensor):
        tensor_clamp = torch.clamp(tensor, 0, 1)
        return tensor + (tensor_clamp - tensor).clone().detach()

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, *, pretrain, network):
        # save_dir = '../experiments/pretrained_models/'
        # if model_path == None:
        #     model_path = self.opt['path']['models']
        # if save_dir is None:
        #     save_filename = '{}_{}_{}.pth'.format(accuracy, iter_label, network_label)
        #     save_path = os.path.join(model_path, save_filename)
        # else:
        #     save_filename = '{}_latest.pth'.format(network_label)
        #     save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        print("Model saved to: {}".format(pretrain))
        torch.save(state_dict, pretrain)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)


    def calculate_pixel_f1(self, pd, gt):
        seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
        true_pos = float(np.logical_and(pd, gt).sum())
        true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
        false_pos = float(np.logical_and(pd, gt_inv).sum())
        false_neg = float(np.logical_and(seg_inv, gt).sum())
        f1 = 2.0 * true_pos / (2.0 * true_pos + false_pos + false_neg + 1e-6)
        # precision = true_pos / (true_pos + false_pos + 1e-6)
        # recall = true_pos / (true_pos + false_neg + 1e-6)
        # accuracy = (true_pos + true_neg) / (false_pos + false_neg + true_pos + true_neg)
        return f1

    ### todo: image manipulations
    def data_augmentation_on_rendered_rgb(self, modified_input, index=None):
        if index is None:
            index = self.global_step % 4

        is_stronger = np.random.rand() > 0.5
        # if index in self.opt['simulated_hue']:
        #     ## careful!
        #     strength = np.random.rand() * (0.05 if is_stronger>0 else -0.05)
        #     modified_adjusted = F.adjust_hue(modified_input, hue_factor=0+strength)  # 0.5 ave
        if index == 0:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = Functional.adjust_contrast(modified_input, contrast_factor=1+strength)  # 1 ave
        elif index == 1:
            ## careful!
            strength = np.random.rand() * (0.05 if is_stronger > 0 else -0.05)
            modified_adjusted = Functional.adjust_gamma(modified_input, gamma=1+strength) # 1 ave
        elif index == 2:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = Functional.adjust_saturation(modified_input, saturation_factor=1+strength)
        elif index == 3:
            strength = np.random.rand() * (0.3 if is_stronger > 0 else -0.3)
            modified_adjusted = Functional.adjust_brightness(modified_input,
                                                    brightness_factor=1+strength)  # 1 ave
        else:
            raise NotImplementedError("图像增强的index错误，请检查！")
        modified_adjusted = self.clamp_with_grad(modified_adjusted)

        return modified_adjusted #modified_input + (modified_adjusted - modified_input).detach()

    def benign_attacks(self, *, forward_image, index, quality_idx=None, kernel_size=None, resize_ratio=None):
        '''
            contains both simulation and real-world attack
            we restrict the lower bound of PSNR by attack, can be modified in setting.
        '''

        if quality_idx is None:
            kernel_size = 7 #random.choice([5, 7])  # 3,5,7
            resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                            int(self.random_float(0.5, 2) * self.width_height))
            # index_for_postprocessing = index #self.global_step
            quality_idx = random.randint(16,21) if index % 3 == 2 else 20

        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.empty_like(forward_image)

        if index is None:
            index = self.global_step
        index = index % 3
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])  # 3,5,7
        if resize_ratio is None:
            resize_ratio = (int(self.random_float(0.7, 1.5) * self.width_height),
                        int(self.random_float(0.7, 1.5) * self.width_height))


        ## id of weak JPEG: 0,1,2,4,6,7
        if index == 0:
            ## resize sometimes will also cause very low PSNR
            blurring_layer = self.resize
            processed_image, resize_ratio = blurring_layer(forward_image, resize_ratio=resize_ratio)
        # elif index in self.opt['simulated_gblur_indices']:
        #     ## additional care for gaussian and median blur
        #     blurring_layer = self.gaussian_blur
        #     processed_image, kernel_size = blurring_layer(attacked_forward, kernel_size=kernel_size)
        # elif index in self.opt['simulated_mblur_indices']:
        #     blurring_layer = self.median_blur
        #     processed_image, kernel_size = blurring_layer(attacked_forward, kernel=kernel_size)
        elif index == 1:
            ## we dont simulate gaussian but direct add
            blurring_layer = self.identity
            processed_image = blurring_layer(forward_image)
        elif index == 2:
            blurring_layer = self.identity
            processed_image = blurring_layer(forward_image)
        else:
            raise NotImplementedError("postprocess的Index没有找到，请检查！")

        ## we regulate that jpeg attack also should not cause PSNR to be lower than 30dB
        jpeg_result = processed_image
        quality = quality_idx * 5
        for q_index in range(quality_idx,21):
            quality = int(q_index * 5)
            jpeg_layer_after_blurring = self.jpeg_simulate[q_index - 10][0] if quality < 100 else self.identity
            jpeg_result = jpeg_layer_after_blurring(processed_image)
            psnr = self.psnr(self.postprocess(jpeg_result), self.postprocess(processed_image)).item()
            if psnr>=self.opt['minimum_PSNR_caused_by_attack']:
                break
        attacked_real_jpeg_simulate = self.clamp_with_grad(jpeg_result)

        ## real-world attack
        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality,
                                                                    kernel=kernel_size, resize_ratio=resize_ratio,
                                                                    index=index)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        attacked_real_jpeg = attacked_real_jpeg.clone().detach()
        attacked_image = attacked_real_jpeg_simulate + (
                    attacked_real_jpeg - attacked_real_jpeg_simulate).clone().detach()

        # error_scratch = attacked_real_jpeg - attacked_forward
        # l_scratch = self.l1_loss(error_scratch, torch.zeros_like(error_scratch).cuda())
        # logs.append(('SCRATCH', l_scratch.item()))
        return attacked_image, attacked_real_jpeg_simulate, (kernel_size, quality_idx, resize_ratio)

    def benign_attacks_without_simulation(self, *, forward_image, index, quality_idx=None, kernel_size=None,
                                                         resize_ratio=None):
        '''
            real-world attack, whose setting should be fed.
        '''

        if quality_idx is None:
            kernel_size = 7 #random.choice([5, 7])  # 3,5,7
            resize_ratio = (int(self.random_float(0.5, 2) * self.width_height),
                            int(self.random_float(0.5, 2) * self.width_height))
            # index_for_postprocessing = index #self.global_step
            quality_idx = random.randint(16,21) if index % 3 == 2 else 20

        batch_size, height_width = forward_image.shape[0], forward_image.shape[2]
        attacked_real_jpeg = torch.empty_like(forward_image)
        quality = int(quality_idx * 5)

        for idx_atkimg in range(batch_size):
            grid = forward_image[idx_atkimg]
            realworld_attack = self.real_world_attacking_on_ndarray(grid=grid, qf_after_blur=quality,
                                                                    index=index, kernel=kernel_size,
                                                                    resize_ratio=resize_ratio)
            attacked_real_jpeg[idx_atkimg:idx_atkimg + 1] = realworld_attack

        return attacked_real_jpeg.cuda()

    def real_world_attacking_on_ndarray(self, *,  grid, qf_after_blur, kernel, resize_ratio, index=None):
        # batch_size, height_width = self.real_H.shape[0], self.real_H.shape[2]
        '''
            real-world attack (CV2)
            ref: https://www.geeksforgeeks.org/python-opencv-imencode-function/
            imencode will produce the exact result compared to that by imwrite, but much quicker
        '''
        if index is None:
            index = self.global_step
        index = index % 3

        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()
        if index ==0:
            # grid = self.resize(grid.unsqueeze(0))[0]
            # newH, newW = int((0.7+0.6*np.random.rand())*self.width_height), int((0.7+0.6*np.random.rand())*self.width_height)
            newH, newW = resize_ratio
            realworld_attack = cv2.resize(np.copy(ndarr), (newH,newW),
                                          interpolation=cv2.INTER_LINEAR)
            realworld_attack = cv2.resize(np.copy(realworld_attack), (self.width_height, self.width_height),
                                interpolation=cv2.INTER_LINEAR)
        # elif index ==1:
        #     # kernel_list = [5]
        #     # kernel = random.choice(kernel_list)
        #     realworld_attack = cv2.GaussianBlur(ndarr, (kernel, kernel), 0) if kernel > 0 else ndarr
        # elif index ==2:
        #     # kernel_list = [5]
        #     # kernel = random.choice(kernel_list)
        #     realworld_attack = cv2.medianBlur(ndarr, kernel) if kernel > 0 else ndarr

        elif index ==1:
            mean, sigma = 0, 1.0
            gauss = np.random.normal(mean, sigma, (self.width_height, self.width_height, 3))
            # 给图片添加高斯噪声
            realworld_attack = ndarr + gauss
        elif index ==2:
            realworld_attack = ndarr
        else:
            raise NotImplementedError("postprocess的Index没有找到，请检查！")

        _, realworld_attack = cv2.imencode('.jpeg', realworld_attack,
                                           (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        realworld_attack = cv2.imdecode(realworld_attack, cv2.IMREAD_UNCHANGED)
        # realworld_attack = data.util.channel_convert(realworld_attack.shape[2], 'RGB', [realworld_attack])[0]
        # realworld_attack = cv2.resize(copy.deepcopy(realworld_attack), (height_width, height_width),
        #                               interpolation=cv2.INTER_LINEAR)

        # ### jpeg in the file
        # cv2.imwrite('./temp.jpeg', realworld_attack,
        #                                    (int(cv2.IMWRITE_JPEG_QUALITY), qf_after_blur))
        # realworld_attack = cv2.imread('./temp.jpeg', cv2.IMREAD_COLOR)
        # realworld_attack = realworld_attack.astype(np.float32) / 255.
        # if realworld_attack.ndim == 2:
        #     realworld_attack = np.expand_dims(realworld_attack, axis=2)
        # # some images have 4 channels
        # if realworld_attack.shape[2] > 3:
        #     realworld_attack = realworld_attack[:, :, :3]
        # orig_height, orig_width, _ = realworld_attack.shape
        # H, W, _ = realworld_attack.shape
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # if realworld_attack.shape[2] == 3:
        #     realworld_attack = realworld_attack[:, :, [2, 1, 0]]

        realworld_attack = realworld_attack.astype(np.float32) / 255.
        realworld_attack = torch.from_numpy(
            np.ascontiguousarray(np.transpose(realworld_attack, (2, 0, 1)))).contiguous().float()
        realworld_attack = realworld_attack.unsqueeze(0)
        return realworld_attack



    ####################################################################################################
    # todo: define how to tamper the rendered RGB
    ####################################################################################################
    def tampering_RAW(self, *, masks_GT, modified_input, tamper_source, percent_range=(0.05,0.25), index=None):
        batch_size, height_width = modified_input.shape[0], modified_input.shape[2]
        ####### Tamper ###############
        # attacked_forward = torch.zeros_like(modified_input)
        # for img_idx in range(batch_size):
        if index is None:
            index = self.global_step % 6

        if index in [0,2]:  # self.using_splicing():
            ### todo: splicing
            attacked_forward = self.splicing(forward_image=modified_input, masks=masks_GT, tamper_source=tamper_source)

        elif index in [1,3,5]:  # self.using_copy_move():
            ### todo: copy-move
            attacked_forward, masks_GT = self.copymove(forward_image=modified_input,
                                                              masks_GT=masks_GT, percent_range=percent_range)
            # del self.tamper_shifted
            # del self.mask_shifted
            # torch.cuda.empty_cache()

        # elif index in self.opt['simulated_copysplicing_indices']:  # self.using_simulated_inpainting:
        #     ### todo: copy-splicing
        #     attacked_forward, masks, masks_GT = self.copysplicing(forward_image=modified_input, masks=masks,
        #                                                           percent_range=percent_range,
        #                                                           another_immunized=self.previous_protected)
        elif index in [4]:  # self.using_splicing():
            ### todo: inpainting
            ## note! self.global_step_for_inpainting, not index, decides which inpainting model will be used

            use_which_inpainting = self.global_step_for_inpainting % 2
            # if use_which_inpainting in self.opt['ideal_as_inpainting']:
            #
            #     ## ideal
            #     attacked_forward_edgeconnect = self.inpainting_for_RAW(forward_image=modified_input, masks=masks,
            #                                                            gt_rgb=gt_rgb)

            # elif use_which_inpainting in self.opt['edgeconnect_as_inpainting']:
            #     ## edgeconnect
            #     attacked_forward_edgeconnect = self.inpainting_edgeconnect(forward_image=modified_input,
            #                                                                masks=masks_GT)
            if use_which_inpainting in [0]:
                ## zits
                attacked_forward_edgeconnect = self.inpainting_ZITS(forward_image=modified_input,
                                                                    masks=masks_GT)
            else:  # if use_which_inpainting in self.opt['lama_as_inpainting']:
                ## lama
                attacked_forward_edgeconnect = self.inpainting_lama(forward_image=modified_input,
                                                                    masks=masks_GT)

            attacked_forward = attacked_forward_edgeconnect
            self.global_step_for_inpainting += 1

        else:
            print(index)
            raise NotImplementedError("Tamper的方法没找到！请检查！")

        attacked_forward = self.clamp_with_grad(attacked_forward)
        # attacked_forward = self.Quantization(attacked_forward)

        return attacked_forward, masks_GT


    def define_inpainting_ZITS(self):
        from inpainting_methods.ZITSinpainting.src.FTR_trainer import ZITSModel
        from shutil import copyfile
        from inpainting_methods.ZITSinpainting.src.config import Config
        print("Building ZITS...........please wait...")
        model_path = '/groupshare/ckpt/zits_places2_hr'
        config_path = os.path.join(model_path, 'config.yml')

        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(config_path):
            copyfile('./ZITSinpainting/config_list/config_ZITS_HR_places2.yml', config_path)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = Config(config_path)
        config.MODE = 1
        # config.GPUS = 1
        # config.GPU_ids = '0'
        # config.world_size = 1
        self.ZITS_model = ZITSModel(config=config, test=True).cuda()
        self.ZITS_model = DistributedDataParallel(self.ZITS_model,
                                                  device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)
        self.ZITS_model.eval()

    def define_inpainting_edgeconnect(self):
        from inpainting_methods.edgeconnect.main import load_config
        from inpainting_methods.edgeconnect.src.models import EdgeModel, InpaintingModel
        print("Building edgeconnect...........please wait...")
        config = load_config(mode=2)
        self.edge_model = EdgeModel(config)
        self.inpainting_model = InpaintingModel(config)
        self.edge_model.load()
        self.inpainting_model.load()
        self.edge_model = self.edge_model.cuda()
        self.inpainting_model = self.inpainting_model.cuda()
        self.edge_model = DistributedDataParallel(self.edge_model,
                                                  device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)
        self.inpainting_model = DistributedDataParallel(self.inpainting_model,
                                                        device_ids=[torch.cuda.current_device()],
                                                        find_unused_parameters=True)
        self.edge_model.eval()
        self.inpainting_model.eval()
        # self.edgeconnect_model = get_model()
        # self.edgeconnect_model = DistributedDataParallel(self.edgeconnect_model,
        #                                               device_ids=[torch.cuda.current_device()],
        #                                               find_unused_parameters=True)


    def define_inpainting_lama(self):
        from inpainting_methods.saicinpainting.training.trainers import load_checkpoint
        print("Building LAMA...........please wait...")
        checkpoint_path = '/groupshare/codes/inpainting_methods/big-lama/models/best.ckpt'
        train_config_path = '/groupshare/codes/inpainting_methods/big-lama/config.yaml'
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # out_ext = predict_config.get('out_ext', '.png')

        self.lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        # if not refine == False:
        # if not predict_config.get('refine', False):
        self.lama_model = self.lama_model.cuda()

    @torch.no_grad()
    def inpainting_lama(self, *, forward_image, masks):
        batch = {
            'image': forward_image,
            'mask': masks
        }
        # batch['mask'] = (batch['mask'] > 0) * 1
        batch = self.lama_model(batch)
        result = batch['inpainted']
        return forward_image * (1 - masks) + result.clone().detach() * masks

    # @torch.no_grad()
    # def inpainting_edgeconnect(self, *, forward_image, masks, image_gray=None, image_canny=None):
    #     # items = (forward_image, image_gray, image_canny, masks)
    #     if image_gray is None:
    #         modified_crop_out = forward_image * (1 - masks)
    #         image_gray, image_canny = self.get_canny(input=modified_crop_out, masks_GT=masks)
    #
    #     self.edge_model.eval()
    #     self.inpainting_model.eval()
    #     edges = self.edge_model(image_gray, image_canny, masks).detach()
    #     outputs = self.inpainting_model(forward_image, edges, masks)
    #     result = (outputs * masks) + forward_image * (1 - masks)
    #
    #     # result = self.edgeconnect_model(items)
    #
    #     return forward_image * (1 - masks) + result.clone().detach() * masks

    def get_canny(self, input, masks_GT=None, sigma=1):
        cannied_list = torch.zeros_like(input)[:,:1]
        gray_list = torch.zeros_like(input)[:,:1]
        for i in range(input.shape[0]):
            grid = input[i]
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to('cpu', torch.uint8).numpy()
            ndarr = ndarr.astype(np.float32) / 255.
            img_gray = rgb2gray(ndarr)
            cannied = canny(img_gray, sigma=sigma, mask=None).astype(np.float32)
            gray_list[i] = torch.from_numpy(np.ascontiguousarray(img_gray)).contiguous().float()
            cannied_list[i] = torch.from_numpy(np.ascontiguousarray(cannied)).contiguous().float()
        return cannied_list, gray_list

    @torch.no_grad()
    def inpainting_ZITS(self, *, forward_image, masks):
        from inpainting_methods.ZITSinpainting.single_image_test import wf_inference_test, load_masked_position_encoding
        sigma = 3.0
        valid_th = 0.85
        # items = load_images_for_test(src_img, mask_img, sigma=sigma)
        ### load_image must be customized
        image_256 = F.interpolate(
            forward_image.clone(),
            size=[256, 256],
            mode='bilinear')
        image_gray, image_canny = self.get_canny(input=image_256, sigma=sigma)

        rel_pos, abs_pos, direct = None, None, None  # torch.zeros_like(masks)[:,0].long(), torch.zeros_like(masks)[:,0].long(), torch.zeros_like(masks)[:,0].long()
        for i in range(forward_image.shape[0]):
            mask_numpy = masks[i, 0].mul(255).add_(0.5).clamp_(0, 255).contiguous().to('cpu', torch.uint8).numpy()
            rel_pos_single, abs_pos_single, direct_single = load_masked_position_encoding(mask_numpy)
            rel_pos_single = torch.LongTensor(rel_pos_single).unsqueeze(0).cuda()
            abs_pos_single = torch.LongTensor(abs_pos_single).unsqueeze(0).cuda()
            direct_single = torch.LongTensor(direct_single).unsqueeze(0).cuda()
            rel_pos = rel_pos_single if rel_pos is None else torch.cat([rel_pos, rel_pos_single], dim=0)
            abs_pos = abs_pos_single if abs_pos is None else torch.cat([abs_pos, abs_pos_single], dim=0)
            direct = direct_single if direct is None else torch.cat([direct, direct_single], dim=0)

        batch = dict()
        batch['image'] = forward_image
        batch['img_256'] = image_256.clone()
        batch['mask'] = masks
        batch['mask_256'] = torch.where(F.interpolate(
            masks.clone(),
            size=[256, 256],
            mode='bilinear') > 0, 1.0, 0.0).cuda()
        batch['mask_512'] = masks.clone()
        batch['edge_256'] = image_canny
        batch['img_512'] = forward_image.clone()
        batch['rel_pos'] = rel_pos
        batch['abs_pos'] = abs_pos
        batch['direct'] = direct
        batch['h'] = forward_image.shape[2]
        batch['w'] = forward_image.shape[3]

        line = wf_inference_test(self.ZITS_model.module.wf, batch['img_512'], h=256, w=256, masks=batch['mask_512'],
                                 valid_th=valid_th, mask_th=valid_th)
        batch['line_256'] = line

        # for k in batch:
        #     if type(batch[k]) is torch.Tensor:
        #         batch[k] = batch[k].cuda()
        merged_image = self.ZITS_model(batch)

        return merged_image.clone().detach() * masks + forward_image * (1 - masks)

    ### todo: image manipulations
    def splicing(self, *, forward_image, masks, tamper_source):
        return forward_image * (1 - masks) + tamper_source * masks

    # def inpainting_for_PAMI(self, *, forward_image, masks, modified_canny):
    #     with torch.no_grad():
    #         reversed_stuff, reverse_feature = self.netG(
    #             torch.cat((forward_image * (1 - masks),
    #                        torch.zeros_like(modified_canny)), dim=1),
    #             rev=True)  # torch.zeros_like(modified_canny).cuda()
    #         reversed_ch1, reversed_ch2 = reversed_stuff[:, :3, :, :], reversed_stuff[:, 3:, :, :]
    #         reversed_image = self.clamp_with_grad(reversed_ch1)
    #         # attacked_forward = forward_image * (1 - masks) + modified_input.clone().detach() * masks
    #         attacked_forward = forward_image * (1 - masks) + reversed_image.clone().detach() * masks
    #         # reversed_image = reversed_image.repeat(way_attack,1,1,1)
    #     del reversed_stuff
    #     del reverse_feature
    #
    #     return attacked_forward

    def get_shifted_image_for_copymove(self, *, forward_image, percent_range, masks):
        batch_size, channels, height_width = forward_image.shape[0], forward_image.shape[1], forward_image.shape[2]
        lower_bound_percent = percent_range[0] + (percent_range[1] - percent_range[0]) * np.random.rand()
        ###### IMPORTANT NOTE: for ideal copy-mopv, here should be forward_image. If you want to ease the condition, can be changed to forward_iamge
        tamper = forward_image.clone().detach()
        max_x_shift, max_y_shift, valid, retried, max_valid, mask_buff = 0, 0, 0, 0, 0, None
        mask_shifted = masks
        while retried <= 20 and valid < lower_bound_percent:
            x_shift = int((height_width) * (0.1 + 0.7 * np.random.rand())) * (1 if np.random.rand() > 0.5 else -1)
            y_shift = int((height_width) * (0.1 + 0.7 * np.random.rand())) * (1 if np.random.rand() > 0.5 else -1)
            # if abs(x_shift) <= (height_width / 4) or abs(y_shift) <= (height_width / 4):
            #     continue
            ### two times padding ###
            mask_buff = torch.zeros((masks.shape[0], masks.shape[1],
                                     masks.shape[2] + abs(2 * x_shift),
                                     masks.shape[3] + abs(2 * y_shift))).cuda()

            mask_buff[:, :,
            abs(x_shift) + x_shift:abs(x_shift) + x_shift + height_width,
            abs(y_shift) + y_shift:abs(y_shift) + y_shift + height_width] = masks

            mask_buff = mask_buff[:, :,
                        abs(x_shift):abs(x_shift) + height_width,
                        abs(y_shift):abs(y_shift) + height_width]

            valid = torch.mean(mask_buff)
            retried += 1
            if valid >= max_valid:
                max_valid = valid
                mask_shifted = mask_buff
                max_x_shift, max_y_shift = x_shift, y_shift

        tamper_shifted = torch.zeros((batch_size, channels,
                                      height_width + abs(2 * max_x_shift),
                                      height_width + abs(2 * max_y_shift))).cuda()
        tamper_shifted[:, :,
        abs(max_x_shift) + max_x_shift: abs(max_x_shift) + max_x_shift + height_width,
        abs(max_y_shift) + max_y_shift: abs(max_y_shift) + max_y_shift + height_width] = tamper

        tamper_shifted = tamper_shifted[:, :,
                         abs(max_x_shift): abs(max_x_shift) + height_width,
                         abs(max_y_shift): abs(max_y_shift) + height_width]

        masks = mask_shifted.clone().detach()

        # masks_GT = masks[:, :1, :, :]

        return tamper_shifted, masks

    def copymove(self, *, forward_image, masks_GT, percent_range):
        batch_size, channels, height_width = forward_image.shape[0], forward_image.shape[1], forward_image.shape[2]
        tamper_shifted, masks_GT = self.get_shifted_image_for_copymove(forward_image=forward_image,
                                                                              percent_range=percent_range,
                                                                              masks=masks_GT)
        attacked_forward = forward_image * (1 - masks_GT) + tamper_shifted.clone().detach() * masks_GT

        return attacked_forward, masks_GT

    # def copysplicing(self, *, forward_image, masks, percent_range, another_immunized=None):
    #     with torch.no_grad():
    #         if another_immunized is None:
    #             another_generated = self.netG(
    #                 torch.cat([self.previous_previous_images, self.previous_previous_canny], dim=1))
    #             another_immunized = another_generated[:, :3, :, :]
    #             another_immunized = self.clamp_with_grad(another_immunized)
    #         tamper_shifted, masks, masks_GT = self.get_shifted_image_for_copymove(forward_image=another_immunized,
    #                                                                               percent_range=percent_range,
    #                                                                               masks=masks)
    #         attacked_forward = forward_image * (1 - masks) + another_immunized.clone().detach() * masks
    #     # del another_generated
    #
    #     return attacked_forward, masks, masks_GT

    # def copysplicing(self, *, forward_image, masks, percent_range, another_immunized=None):
    #     with torch.no_grad():
    #         if another_immunized is None:
    #             another_generated = self.netG(
    #                 torch.cat([self.previous_previous_images, self.previous_previous_canny], dim=1))
    #             another_immunized = another_generated[:, :3, :, :]
    #             another_immunized = self.clamp_with_grad(another_immunized)
    #         tamper_shifted, masks, masks_GT = self.get_shifted_image_for_copymove(forward_image=another_immunized,
    #                                                                               percent_range=percent_range,
    #                                                                               masks=masks)
    #         attacked_forward = forward_image * (1 - masks) + another_immunized.clone().detach() * masks
    #     # del another_generated
    #
    #     return attacked_forward, masks, masks_GT

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


if __name__ == '__main__':
    pass