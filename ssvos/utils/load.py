import torch
import os

def load_checkpoint(model, ckpt_path, map_location='cpu', finetune=False, logger=None, accelerator=None, **kwargs):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if finetune:
        start_epoch = 1
        load_encoder(model, ckpt['state_dict'], **kwargs)
        if accelerator.is_main_process:
            logger.info("Finetuning ...")
    else:
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']+1
        if accelerator.is_main_process:
            logger.info(f"Resume from epoch {start_epoch-1}...")
    return start_epoch

def load_encoder(model, state_dict, **kwargs):
    pretrained_encoder_key = kwargs['pretrained_encoder_key']
    model_encoder_key = kwargs['model_encoder_key']
    encoder_dict = {}
    for k, v in state_dict.items():
        if pretrained_encoder_key in k:
            start = len(pretrained_encoder_key)+1
            encoder_dict[k[start:]] = v
    assert hasattr(model, model_encoder_key)
    encoder = getattr(model, model_encoder_key)
    encoder.load_state_dict(encoder_dict, strict=False)

def load_pretrained_weights(model, pretrained_weights, checkpoint_key='teacher', model_name='deit_small', patch_size=8):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "deit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "deit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")
    