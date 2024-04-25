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
from .models_mae import MaskedAutoencoderViT


class MaskedAutoencoderViT_Depth(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim=1024, **kwargs):
        super().__init__(embed_dim=embed_dim, **kwargs)

    def forward_encoder(self, x_dep):
        # embed patches
        x_dep = self.patch_embed(x_dep)

        # add pos embed w/o cls token
        x_dep = x_dep + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token_dep = self.cls_token_dep + self.pos_embed[:, :1, :]
        cls_tokens_dep = cls_token_dep.expand(x_dep.shape[0], -1, -1)
        x_dep = torch.cat((cls_tokens_dep, x_dep), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x_dep = blk(x_dep)
        x_dep = self.norm(x_dep)

        return x_dep

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x_dep = self.decoder_pred(x)

        # remove cls token
        x_dep = x_dep[:, 1:, :]

        return x_dep

    def forward_loss(self, imgs_dep, pred_dep):
        N = imgs_dep.size(0)
        target_dep = self.patchify(imgs_dep)
        loss_dep = (pred_dep - target_dep) ** 2
        loss_dep = loss_dep.mean(dim = -1)
        loss_dep = loss_dep.sum() / N

        loss = {'dep_loss': loss_dep}
        return loss

    def forward(self, imgs_dep_masked, imgs_dep=None):
        latent = self.forward_encoder(imgs_dep_masked)
        pred_dep = self.forward_decoder(latent)  # [N, L, p*p*3]
        if imgs_dep is None:
            return pred_dep
        loss = self.forward_loss(imgs_dep, pred_dep)
        return loss


def mae_vit_base_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT_Depth(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT_Depth(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(patch_size=14, **kwargs):
    model = MaskedAutoencoderViT_Depth(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_depth_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_depth_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_depth_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
