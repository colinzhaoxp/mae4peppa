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


class MaskedAutoencoderViT_Weight(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim=1024, class_num=1, **kwargs):
        super().__init__(embed_dim=embed_dim, class_num=class_num, **kwargs)

        # --------------------------------------------------------------------------
        self.weight_pred = nn.Sequential(
            nn.Linear(embed_dim, class_num, bias=False)
        )
        # --------------------------------------------------------------------------

    def forward_encoder(self, x):
        x, x_dep = x
        # embed patches
        x = self.patch_embed(x)
        x_dep = self.patch_embed_dep(x_dep)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        x_dep = x_dep + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_token_dep = self.cls_token_dep + self.pos_embed[:, :1, :]
        cls_tokens_dep = cls_token_dep.expand(x_dep.shape[0], -1, -1)
        x_dep = torch.cat((cls_tokens_dep, x_dep), dim=1)

        # fusion rgb feats and depth feats
        fusion_feats = self.fusion(x, x_dep)

        # apply Transformer blocks
        for blk in self.blocks:
            fusion_feats = blk(fusion_feats)
        fusion_feats = self.norm(fusion_feats)

        cls = fusion_feats[:, 0, :] # batch * 512
        pred = self.weight_pred(cls)

        return fusion_feats, pred

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x_rgb = self.decoder_pred(x)
        x_dep = self.decoder_pred_dep(x)

        # remove cls token
        x_rgb = x_rgb[:, 1:, :]
        x_dep = x_dep[:, 1:, :]

        return x_rgb, x_dep

    def forward(self, imgs_masked, imgs=None):
        latent, weight_pred = self.forward_encoder(imgs_masked)
        pred, pred_dep = self.forward_decoder(latent)  # [N, L, p*p*3]
        if imgs is None:
            return pred, pred_dep, weight_pred
        loss = self.forward_loss(imgs, (pred, pred_dep))
        return loss, weight_pred


def mae_vit_base_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT_Weight(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT_Weight(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(patch_size=14, **kwargs):
    model = MaskedAutoencoderViT_Weight(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_weight_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_weight_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_weight_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
