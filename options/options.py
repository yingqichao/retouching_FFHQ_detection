import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()



def parse(*, opt_path,
          base_opt_path=None,
          attack_opt_path=None,
          args=None):

    # args.task_name = args.task_name.lower()
    # model_and_size = args.task_name.split("_")
    # if len(model_and_size)<3:
    #     raise NotImplementedError("请在task_name里面写清楚训练策略，网络结构和图像大小，格式样例：Hallucinate_ViT_224")

    ## base option ##
    if base_opt_path is None:
        base_opt_path = 'options/meiyan_init_option.yml'
    with open(base_opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    ## attack layer option ##
    if attack_opt_path is None:
        attack_opt_path = 'options/attack_layer.yml'
    with open(attack_opt_path, mode='r') as f:
        opt_attack = yaml.load(f, Loader=Loader)
    opt.update(opt_attack)

    ## local local custom option ##
    with open(opt_path, mode='r') as f:
        opt_new = yaml.load(f, Loader=Loader)

    opt.update(opt_new)

    ## todo: add default values if not specified in cunstom yml

    print(f"Opt List: {opt}")

    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)


    # SIZE = int(model_and_size[2])
    # if "tres" in model_and_size[1] or "vit" in model_and_size[1]:
    #     SIZE = 224
    # opt['datasets']['train']['GT_size'] = SIZE
    # opt['network_arch'] = model_and_size[1]
    opt['methodology'] = "meiyan" #model_and_size[0]

    print(f"train_dataset: {opt['train_dataset']}")
    print(f"test_dataset: {opt['test_dataset']}")
    print(f"with_data_aug: {opt['with_data_aug']}")

    if 'save_checkpoint' not in opt:
        opt['save_checkpoint'] = opt['conduct_train']


    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['models']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
