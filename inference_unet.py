import torch


if __name__ == '__main__':
    from models.vit_topdown.configs.config import get_cfg
    from models.vit_topdown.vit_top_down import vit_topdown_base_patch16_224
    cfg = get_cfg()
    if "prompt" in cfg.MODEL.TRANSFER_TYPE:
        print("prompt config loaded! ")
        prompt_cfg = cfg.MODEL.PROMPT
        print(prompt_cfg)
    else:
        prompt_cfg = None
    detection_model, feat_dim = vit_topdown_base_patch16_224(pretrained=False, cfg=cfg, prompt_cfg=prompt_cfg,
                                                                       drop_path_rate=0.1)
    input = torch.ones((1,3,224,224))
    output = detection_model(input)