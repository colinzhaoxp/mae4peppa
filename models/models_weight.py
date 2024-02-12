# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.my_utils import load_checkpoint
from .fusion import Fusion

class EncoderViT2weight(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, class_num=1):
        super().__init__()
        self.in_chans = in_chans
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.patch_embed_dep = PatchEmbed(img_size, patch_size, 1, embed_dim)
        num_patches_dep = self.patch_embed_dep.num_patches

        self.fusion = Fusion(embed_dim, heads=num_heads // 2)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # predict weight
        self.weight_pred = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(embed_dim, class_num, bias=False)
        )

        # self.posture_pred = nn.Sequential(
        #     nn.Linear(embed_dim, 3, bias=False)
        # )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3 or 4, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        in_channels = imgs.shape[1]

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_channels))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * in_channels)
        imgs: (N, in_channels, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        in_channels = self.in_chans

        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_channels, h * p, h * p))
        return imgs

    def forward_encoder(self, x, x_dep):
        # embed patches
        x = self.patch_embed(x)
        x_dep = self.patch_embed_dep(x_dep)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_dep = x_dep + self.pos_embed[:, 1:, :]

        # fusion rgb feats and depth feats
        fusion_feats = self.fusion(x, x_dep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        fusion_feats = torch.cat((cls_tokens, fusion_feats), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            fusion_feats = blk(fusion_feats)
        fusion_feats = self.norm(fusion_feats)

        return fusion_feats

    def forward_weight(self, x):
        # x: B * L * N
        x = x[:, 0, :]
        weight_pred = self.weight_pred(x)
        return weight_pred

    def forward_posture(self, x):
        x = x[:, 0, :]
        posture_pred = self.posture_pred(x)
        return posture_pred

    def forward(self, imgs, img_dep=None, mask_ratio=0.75):
        latent = self.forward_encoder(imgs, img_dep)
        weight_pred = self.forward_weight(latent)  # [N, L, p*p*3]
        # pos_pred = self.forward_posture(latent)
        return weight_pred


def mae_vit_base_patch16_dec512d8b(patch_size=16, **kwargs):
    model = EncoderViT2weight(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(patch_size=16, **kwargs):
    model = EncoderViT2weight(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(patch_size=14, **kwargs):
    model = EncoderViT2weight(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_weight_vit_tiny(patch_size=2, **kwargs):
    model = EncoderViT2weight(
        patch_size=patch_size, embed_dim=256, depth=4, num_heads=4, **kwargs)
    return model


# set recommended archs
mae_weight_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_weight_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_weight_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks