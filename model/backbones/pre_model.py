import json
import os
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
import torch
from typing import Callable, List, Tuple
from model.utils.extraction import instantiate_extractor
from model.vitac.v_repic import V_RePic
from model.vitac.vtt_reall import VTT_ReAll
from model.vitac.t_retac import T_ReTac
from model.vitac.t_retacv1 import T_ReTac as T_ReTacV1


import torchvision.transforms as T
# from torchvision.models import get_model, list_models, ResNet
import torchvision.models as models

if os.path.exists("model/backbones/pre_model_baselines"):
    from model.backbones.pre_model_baselines import clip
    from model.backbones.pre_model_baselines import voltron
    from model.backbones.pre_model_baselines import r3m
    from model.backbones.pre_model_baselines import mvp

from einops import rearrange
import numpy as np
DEFAULT_CACHE = "cache/"
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
class ExtractorMLP(nn.Module):
    def __init__(self, in_dim, out_dim, input_type):
        super(ExtractorMLP, self).__init__()
        self.layers = nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_dim)
                )
        if input_type=='v':
            self.i_feat = nn.Linear(out_dim*196, 128)
        elif input_type=='t20':
            self.t_feat = nn.Linear(out_dim*20, 128)
        elif input_type=='vt20t':
            self.i_feat = nn.Linear(out_dim * 196, 128)
            self.t_feat = nn.Linear(out_dim * 20, 128)
        self.input_type = input_type
    def forward(self, x):
        if self.input_type=='v':
            feat = self.i_feat(x)
        elif self.input_type=='t20':
            feat = self.t_feat(x)
        elif self.input_type=='vt20t':
            i_feat = self.i_feat(x[:, 196].view(x.shape[0], -1))
            t_feat = self.t_feat(x[:, 196:].view(x.shape[0], -1))
            feat = torch.cat([i_feat, t_feat], dim=-1)
        else:
            raise AssertionError
        return feat
class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls', f_ex_mode='MAP'):
        super(Encoder, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone, self.preprocess, gap_dim = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode == 'patch':
            if f_ex_mode=='MAP':
                self.projector = nn.Sequential(
                    instantiate_extractor(self.backbone, n_latents=1)(),
                    nn.Linear(gap_dim, emb_dim))
            elif f_ex_mode=='MLP':
                self.projector = ExtractorMLP(gap_dim, 8, model_name.split('-')[0])
            else:
                raise AssertionError
        else:
            self.projector = nn.Linear(gap_dim, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.freeze = freeze
        self.f_ex_mode = f_ex_mode

    @torch.no_grad()
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            # self.save_img(x[0])
            x = self.preprocess(x)
            if self.model_name == "Voltron":
                feat = self.backbone.__call__(x, mode="visual")
                # p_feat_img = self.projector_img(feat_img)
                # p_feat_tac = self.projector_tac(tac)
                # return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat(
                #     [feat_img, tac.unsqueeze(-1).repeat(1, 1, feat_img.shape[-1])], dim=1)
            elif self.model_name in {"CLIP", "R3M", "MVP"}:
                feat = self.backbone.__call__(x)
                # p_feat_img = self.projector_img(feat_img)
                # p_feat_tac = self.projector_tac(tac)
                # return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac], dim=1)
            elif self.model_name in {"resnet18"}:
                feat = self.backbone.__call__(x)
                feat = torch.flatten(feat, 1)
                # p_feat_img = self.projector_img(feat_img)
                # p_feat_tac = self.projector_tac(tac)
                # return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac], dim=1)
            else:
                feat = self.backbone.get_representations(x, mode=self.en_mode)
        elif isinstance(x, tuple):
            imag, tac = x
            imag = self.preprocess(imag)
            feat = self.backbone.get_representations(imag, tac, mode=self.en_mode)
        else:
            raise AttributeError(f'type of input {x.type} is not expected')
        if self.f_ex_mode=='MeanPolling':
            feat = feat.mean(dim=1)
        return self.projector(feat), feat

    def forward_feat(self, feat):
        return self.projector(feat)

    def save_img(self, img):
        from torchvision import transforms
        from PIL import Image
        import torch

        img_pil = transforms.ToPILImage()(img)

        img_pil.save("saved_image.jpg")


    def visual_backbone(self, x):
        loss, recons, _ = self.backbone(x)
        recon_imgs = self.backbone.generate_origin_img(recons.cpu(), x.cpu())
        save_recons_imgs(x.cpu(), recon_imgs.cpu(),
                         Path(self.pretrain_dir),
                         identify=f"train_{self.model_name}",
                         online_normalization=NORMALIZATION)

class Encoder2(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls'):
        super(Encoder2, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        assert isinstance(model_name, list)
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone_img, self.preprocess, gap_dim_img = load(model_id=model_name[0], freeze=freeze, cache=pretrain_dir)
        self.backbone_tac, _, gap_dim_tac = load(model_id=model_name[1], freeze=freeze, cache=pretrain_dir)
        if self.en_mode != 'cls':
            self.projector_img = nn.Sequential(
                instantiate_extractor(self.backbone_img, n_latents=1)(),
                nn.Linear(gap_dim_img, emb_dim))
            self.projector_tac = nn.Sequential(
                instantiate_extractor(self.backbone_tac, n_latents=1)(),
                nn.Linear(gap_dim_tac, emb_dim))
        else:
            self.projector_img = nn.Linear(gap_dim_img, emb_dim)
            self.projector_tac = nn.Linear(gap_dim_tac, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.emb_dim = emb_dim
        self.freeze = freeze


    @torch.no_grad()
    def forward(self, x):
        # if isinstance(x, torch.Tensor):
        #     x = self.preprocess(x)
        #     feat = self.backbone.get_representations(x, mode=self.en_mode)
        if isinstance(x, tuple):
            imag, tac = x
            imag = self.preprocess(imag)
            feat_img = self.backbone_img.get_representations(imag, mode=self.en_mode)
            p_feat_img = self.projector_img(feat_img)
            feat_tac = self.backbone_tac.get_representations(tac, mode=self.en_mode)
            p_feat_tac = self.projector_tac(feat_tac)
        else:
            raise AttributeError(f'type of input {x.type} is not expected')

        return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, feat_tac], dim=1)

    def forward_feat(self, feat):
        feat_img, feat_tac = feat[:, :384], feat[:, 384:]
        p_feat_img = self.projector_img(feat_img)
        p_feat_tac = self.projector_tac(feat_tac)
        return torch.cat([p_feat_img, p_feat_tac], dim=-1)

    def visual_backbone(self, x):
        loss, recons, _ = self.backbone(x)
        recon_imgs = self.backbone.generate_origin_img(recons.cpu(), x.cpu())
        save_recons_imgs(x.cpu(), recon_imgs.cpu(),
                         Path(self.pretrain_dir),
                         identify=f"train_{self.model_name[0]}",
                         online_normalization=NORMALIZATION)
class EncoderVE_T(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls'):
        super(EncoderVE_T, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        # assert isinstance(model_name, list)
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone_img, self.preprocess, gap_dim_img = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode != 'cls':
            self.projector_img = nn.Sequential(
                instantiate_extractor(self.backbone_img, n_latents=1)(),
                nn.Linear(gap_dim_img, emb_dim))
            self.projector_tac = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, emb_dim))
        else:
            self.projector_img = nn.Linear(gap_dim_img, emb_dim)
            self.projector_tac = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, emb_dim))
        # if freeze:
        #     self.backbone.freeze()
        self.emb_dim = emb_dim

        # if isinstance(x, torch.Tensor):
        self.freeze = freeze


    @torch.no_grad()
    def forward(self, x):
        #     x = self.preprocess(x)
        #     feat = self.backbone.get_representations(x, mode=self.en_mode)

        imag, tac = x
        imag = self.preprocess(imag)
        if self.model_name=="Voltron":
            feat_img = self.backbone_img.__call__(imag, mode="visual")
            p_feat_img = self.projector_img(feat_img)
            p_feat_tac = self.projector_tac(tac)
            return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac.unsqueeze(-1).repeat(1, 1, feat_img.shape[-1])], dim=1)
        elif self.model_name in {"CLIP","R3M", "MVP"}:
            feat_img = self.backbone_img.__call__(imag)
            p_feat_img = self.projector_img(feat_img)
            p_feat_tac = self.projector_tac(tac)
            return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac], dim=1)
        elif self.model_name in {"resnet18"}:
            feat_img = self.backbone_img.__call__(imag)
            feat_img = torch.flatten(feat_img, 1)
            p_feat_img = self.projector_img(feat_img)
            p_feat_tac = self.projector_tac(tac)
            return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac], dim=1)
        else:
            raise AssertionError(f"Error Model Name {self.model_name}")



        return torch.cat([p_feat_img, p_feat_tac], dim=-1), torch.cat([feat_img, tac], dim=1)

    def forward_feat(self, feat):
        if self.model_name == "Voltron":
            feat_img, feat_tac = feat[:, :-20, ], feat[:, -20:, 0]
        else:
            feat_img, feat_tac = feat[:, :-20], feat[:, -20:]
        p_feat_img = self.projector_img(feat_img)
        p_feat_tac = self.projector_tac(feat_tac)
        return torch.cat([p_feat_img, p_feat_tac], dim=-1)
class Encoder_T(nn.Module):
    def __init__(self, model_name, pretrain_dir, freeze, emb_dim, en_mode='cls', f_ex_mode='MAP'):
        super(Encoder_T, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.pretrain_dir = pretrain_dir
        self.model_name = model_name
        self.en_mode = en_mode
        self.backbone, _, gap_dim = load(model_id=model_name, freeze=freeze, cache=pretrain_dir)
        if self.en_mode == 'patch':
            self.projector = nn.Sequential(
                instantiate_extractor(self.backbone, n_latents=1)(),
                nn.Linear(gap_dim, emb_dim))
        else:
            self.projector = nn.Linear(gap_dim, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        self.freeze = freeze
        # self.using_mp = using_mp

    @torch.no_grad()
    def forward(self, x):

        feat = self.backbone.get_representations(x, mode=self.en_mode)
        # if self.using_mp:
        #     feat = feat.mean(dim=1)
        return self.projector(feat), feat

    def forward_feat(self, feat):
        return self.projector(feat)

class Encoder_no_pre(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(Encoder_no_pre, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        # assert self.model_name in list_models(), f"{self.model_name} is not included in {list_models()}"
        # self.backbone = get_model(model_name) # using no weights

        if self.model_name == "resnet18":
            self.backbone = models.resnet18()
        else:
            raise AssertionError('Error model names!')
        gap_dim = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # abandon the last two layers

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(gap_dim, emb_dim)
        self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        if isinstance(x, torch.Tensor):
            imag, tac = x[:, :-20], x[:, -20:]
            imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        elif isinstance(x, tuple):
            imag, tac = x
        else:
            raise AssertionError
        imag = self.preprocess(imag)
        img_feat = self.backbone(imag)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.projector_img(img_feat)
        tac_feat = self.projector_tac(tac)

        feat = torch.cat([img_feat, tac_feat], dim=-1)


        return feat
class EncoderV_no_pre(nn.Module):

    def __init__(self, model_name, emb_dim):
        super(EncoderV_no_pre, self).__init__()
        # assert model_name in _MODELS, f"Unknown model name {model_name}"
        # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        # img_size = 256 if "-256-" in model_name else 224
        self.model_name = model_name
        # self.en_mode = en_mode
        # assert self.model_name in list_models(), f"{self.model_name} is not included in {list_models()}"
        # self.backbone = get_model(model_name) # using no weights

        if self.model_name == "resnet18":
            self.backbone = models.resnet18()
        else:
            raise AssertionError('Error model names!')
        gap_dim = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # abandon the last two layers

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ])

        self.projector_img = nn.Linear(gap_dim, emb_dim)
        # self.projector_tac = nn.Linear(20, emb_dim)
        # if freeze:
        #     self.backbone.freeze()
        # self.freeze = freeze
    def forward(self, x):

        # if isinstance(x, torch.Tensor):
        #     imag, tac = x[:, :-20], x[:, -20:]
        #     imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
        # elif isinstance(x, tuple):
        #     imag, tac = x
        # else:
        #     raise AssertionError
        x = self.preprocess(x)
        img_feat = self.backbone(x)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.projector_img(img_feat)
        # tac_feat = self.projector_tac(tac)

        # feat = torch.cat([img_feat, tac_feat], dim=-1)


        return img_feat

# class Encoder_CNN(nn.Module):
#
#     def __init__(self, model_name, emb_dim):
#         super(Encoder_CNN, self).__init__()
#         # assert model_name in _MODELS, f"Unknown model name {model_name}"
#         # model_func = _MODEL_FUNCS[model_name.split("-")[0]]
#         # img_size = 256 if "-256-" in model_name else 224
#         self.model_name = model_name
#         # self.en_mode = en_mode
#         assert self.model_name =='CNN', f"{self.model_name} is not CNN"
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#
#         self.preprocess = T.Compose(
#             [
#                 T.ConvertImageDtype(torch.float),
#                 T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
#             ])
#
#         self.projector_img = nn.Linear(64*56*56, emb_dim)
#         self.projector_tac = nn.Linear(20, emb_dim)
#         # if freeze:
#         #     self.backbone.freeze()
#         # self.freeze = freeze
#     def forward(self, x):
#
#         if isinstance(x, torch.Tensor):
#             imag, tac = x[:, :-20], x[:, -20:]
#             imag = imag.view(-1, 224, 224, 3).permute(0, 3, 1, 2).to(torch.uint8)  # image
#         elif isinstance(x, tuple):
#             imag, tac = x
#         else:
#             raise AssertionError
#         imag = self.preprocess(imag)
#         img_feat = self.relu(self.conv1(imag))
#         img_feat = self.maxpool(img_feat)
#         img_feat = self.relu(self.conv2(img_feat))
#         img_feat = self.maxpool(img_feat)
#         img_feat = img_feat.contiguous().view(img_feat.size(0), -1)
#
#         img_feat = self.projector_img(img_feat)
#         tac_feat = self.projector_tac(tac)
#
#         feat = torch.cat([img_feat, tac_feat], dim=-1)
#
#
#         return feat


MODEL_REGISTRY = {

    "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310": {
        "config": "model/vitac/model_and_config/t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310.json",
        "checkpoint": "model/vitac/model_and_config/t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310.pt",
        "cls": T_ReTacV1,
    },
    "v-repic-bin-ft-cls+dataset-ViTacReal-all-170": {
        "config": "model/vitac/model_and_config/v-repic-bin-ft-cls+dataset-ViTacReal-all-170.json",
        "checkpoint": "model/vitac/model_and_config/v-repic-bin-ft-cls+dataset-ViTacReal-all-170.pt",
        "cls": V_RePic,
    },
    "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210": {
        "config": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210.json",
        "checkpoint": "model/vitac/model_and_config/vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210.pt",
        "cls": VTT_ReAll,
    },

    "CLIP": {
        "config": "model/backbones/pre_model_baselines/clip/ViT-B-16.pt", #no matter, whatever
        "checkpoint": "model/backbones/pre_model_baselines/clip/ViT-B-16.pt",
        "cls": None,
    },
    "Voltron": {
        "config": "model/backbones/pre_model_baselines/voltron/models/v-cond+vit-small+sth-sth.json",
        "checkpoint": "model/backbones/pre_model_baselines/voltron/models/v-cond+vit-small+sth-sth+epoch-400.pt",
        "cls": None
    },
    "R3M": {
        "config": "model/backbones/pre_model_baselines/r3m/r3m_18/config.yaml",
        "checkpoint": "model/backbones/pre_model_baselines/r3m/r3m_18/model.pt",
        "cls": None
    },
    "MVP": {
        "config": "model/backbones/pre_model_baselines/mvp/mae_pretrain_egosoup_vit_base.pth",
        "checkpoint": "model/backbones/pre_model_baselines/mvp/mae_pretrain_egosoup_vit_base.pth",
        "cls": None
    },

    "resnet18": {
        "config": "model/backbones/pre_model_baselines/resnet18/resnet18-5c106cde.pth",
        "checkpoint": "model/backbones/pre_model_baselines/resnet18/resnet18-5c106cde.pth",
        "cls": None
    },

}
def load(model_id: str, freeze: bool = True, cache: str = DEFAULT_CACHE, device: torch.device = "cpu"):
    """
    Download & cache specified model configuration & checkpoint, then load & return module & image processor.

    Note :: We *override* the default `forward()` method of each of the respective model classes with the
            `extract_features` method --> by default passing "NULL" language for any language-conditioned models.
            This can be overridden either by passing in language (as a `str) or by invoking the corresponding methods.
    """
    assert model_id in MODEL_REGISTRY, f"Model ID `{model_id}` not valid, try one of  {list(MODEL_REGISTRY.keys())}"
    print(f'Load Pre-trained model ---> {model_id}')
    # Download Config & Checkpoint (if not in cache)
    # model_cache = Path(cache) / model_id
    config_path, checkpoint_path = Path(f"{MODEL_REGISTRY[model_id]['config']}"), Path(f"{MODEL_REGISTRY[model_id]['checkpoint']}")
    # os.makedirs(model_cache, exist_ok=True)
    assert checkpoint_path.exists() and config_path.exists(), f'{checkpoint_path} or {config_path} model path does not exist'
    # if not checkpoint_path.exists() or not config_path.exists():
    #     gdown.download(id=MODEL_REGISTRY[model_id]["config"], output=str(config_path), quiet=False)
    #     gdown.download(id=MODEL_REGISTRY[model_id]["checkpoint"], output=str(checkpoint_path), quiet=False)
    if model_id=="CLIP":
        model, _ = clip.load(checkpoint_path, device=device)
        model.__call__ = model.encode_image
        emb_dim = model.visual.output_dim
    elif model_id=="Voltron":
        model, _, emb_dim = voltron.load(model_id, device=device, freeze=freeze)
        # emb_dim = model.a
    elif model_id=="R3M":
        model = r3m.load_r3m('resnet18').eval().to(device)
        emb_dim = 512
    elif model_id=="MVP":
        model, emb_dim = mvp.load("vitb-mae-egosoup", MODEL_REGISTRY[model_id]["checkpoint"])
    elif model_id=="resnet18":
        model = models.resnet18()
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        emb_dim = list(model.children())[-1].in_features
        model = nn.Sequential(*list(model.children())[:-1])  # abandon the last two layers
    else:
        # Load Configuration --> patch `hf_cache` key if present (don't download to random locations on filesystem)
        with open(config_path, "r") as f:
            model_kwargs = json.load(f)
            # if "hf_cache" in model_kwargs:
            #     model_kwargs["hf_cache"] = str(Path(cache) / "hf-cache")

        # By default, the model's `__call__` method defaults to `forward` --> for downstream applications, override!
        #   > Switch `__call__` to `get_representations`
        # MODEL_REGISTRY[model_id]["cls"].__call__ = MODEL_REGISTRY[model_id]["cls"].get_representations

        # Materialize Model (load weights from checkpoint; note that unused element `_` are the optimizer states...)
        model = MODEL_REGISTRY[model_id]["cls"](**model_kwargs)
        if model_id in ['VMVP']:
            state_dict, _ = torch.load(checkpoint_path, map_location=device)
            emb_dim = model_kwargs['encoder_embed_dim']
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
            emb_dim = model_kwargs['encoder_decoder_cfg']['encoder_embed_dim'] if 'vr3m' not in model_id else model_kwargs['encoder_decoder_cfg']['embed_dim']
            model_dict = model.state_dict()
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.replace('module.', '') in model_dict:
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

    # Freeze model parameters if specified (default: True)
    if freeze:
        for _, param in model.named_parameters():
            param.requires_grad = False

    # Build Visual Preprocessing Transform (assumes image is read into a torch.Tensor, but can be adapted)
    if model_id in list(MODEL_REGISTRY.keys()):
        # All models except R3M are by default normalized subject to default IN1K normalization...
        if model_id == "R3M":
            preprocess = T.Compose(
                [
                    # T.Resize(model_kwargs["resolution"]),
                    # T.CenterCrop(model_kwargs["resolution"]),
                    T.ConvertImageDtype(torch.float),
                    T.Lambda(lambda x: x * 255.0),
                ]
            )
        else:
            preprocess = T.Compose(
                [
                    # T.Resize(model_kwargs["dataset_cfg"]["resolution"]),
                    # T.CenterCrop(model_kwargs["resolution"]),
                    T.ConvertImageDtype(torch.float),
                    T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
                ]
            )
    else:
        raise AttributeError(F'{model_id} dose not exit')

    return model, preprocess, emb_dim
    # return model, emb_dim

def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())

def save_recons_imgs(
    ori_imgs: torch.Tensor,
    recons_imgs: torch.Tensor,
    save_dir_input: Path,
    identify: str,
    online_normalization
) -> None:
    # import cv2
    import cv2
    # de online transforms function
    def de_online_transform(ori_img, recon_img, norm=online_normalization):
        # rearrange
        ori_imgs = rearrange(ori_img,"c h w -> h w c")
        recon_imgs = rearrange(recon_img, "c h w -> h w c")
        # to Numpy format
        ori_imgs = ori_imgs.detach().numpy()
        recon_imgs = recon_imgs.detach().numpy()
        # DeNormalize
        ori_imgs = np.array(norm[0]) + ori_imgs * np.array(norm[1])
        recon_imgs = np.array(norm[0]) + recon_imgs * np.array(norm[1])
        # to cv format
        ori_imgs = np.uint8(ori_imgs * 255)
        recon_imgs = np.uint8(recon_imgs * 255)

        return ori_imgs, recon_imgs

    save_dir = save_dir_input / identify
    os.makedirs(str(save_dir), exist_ok=True)
    for bid in range(ori_imgs.shape[0]):
        ori_img = ori_imgs[bid]
        recon_img = recons_imgs[bid]
        # de online norm
        ori_img, recon_img = de_online_transform(ori_img, recon_img)

        ori_save_path = save_dir / f"{bid}_raw.jpg"
        recon_save_path = save_dir / f"{bid}_recon.jpg"

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(ori_save_path), ori_img)
        cv2.imwrite(str(recon_save_path), recon_img)