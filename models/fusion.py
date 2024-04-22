import torch
import torch.nn as nn

"""
对rgb特征和depth特征进行融合
"""

class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Add_Norm(nn.Module):
    def __init__(self, in_dim):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x, sub_output):
        x = self.dropout(x + sub_output)
        x = self.ln(x)
        return x


class Fusion(nn.Module):
    def __init__(self, embed_dim=1024, heads=6):
        super(Fusion, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads

        self.ln_norm_rgb = nn.LayerNorm(self.embed_dim)
        self.ln_norm_dep = nn.LayerNorm(self.embed_dim)
        self.ln_norm = nn.LayerNorm(self.embed_dim)

        self.cross_attention_rgb = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.cross_attention_dep = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.ffn_rgb = FFN(self.embed_dim, self.embed_dim)
        self.ffn_dep = FFN(self.embed_dim, self.embed_dim)
        self.add_norm_rgb = Add_Norm(self.embed_dim)
        self.add_norm_dep = Add_Norm(self.embed_dim)

        self.cross_attention_rgb2d = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.cross_attention_d2rgb = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.ffn_rgb2d = FFN(self.embed_dim, self.embed_dim)
        self.ffn_d2rgb = FFN(self.embed_dim, self.embed_dim)
        self.add_norm_rgb2d = Add_Norm(self.embed_dim)
        self.add_norm_d2rgb = Add_Norm(self.embed_dim)

        self.fusion_conv = nn.Sequential(
            nn.Conv1d(197, 197, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_features, dep_features):
        rgb_feats_norm = self.ln_norm_rgb(rgb_features)
        dep_feats_norm = self.ln_norm_dep(dep_features)

        # self_attention
        rgb_feats_norm = self.cross_attention_rgb(
            rgb_feats_norm, rgb_feats_norm, rgb_feats_norm
        )[0]
        rgb_feats_norm = self.add_norm_rgb(rgb_features, rgb_feats_norm)
        rgb_feats_norm = self.add_norm_rgb(rgb_features_norm, self.ffn_rgb(rgb_feats_norm))

        dep_feats_norm = self.cross_attention_dep(
            dep_feats_norm, dep_feats_norm, dep_feats_norm
        )[0]
        dep_feats_norm = self.add_norm_dep(dep_features, dep_feats_norm)
        dep_feats_norm = self.add_norm_dep(dep_features_norm, self.ffn_dep(dep_feats_norm))

        # cross_attention
        fusion_feats_rgb2d = self.cross_attention_rgb2d(
            rgb_feats_norm, dep_feats_norm, dep_feats_norm
        )[0]
        fusion_feats_rgb2d = self.add_norm_rgb2d(fusion_feats_rgb2d, self.ffn_rgb2d(fusion_feats_rgb2d))

        fusion_feats_d2rgb = self.cross_attention_d2rgb(
            dep_feats_norm, rgb_feats_norm, rgb_feats_norm
        )[0]
        fusion_feats_d2rgb = self.add_norm_d2rgb(fusion_feats_d2rgb, self.ffn_d2rgb(fusion_feats_d2rgb))

        fusion_feats = fusion_feats_rgb2d + fusion_feats_d2rgb
        fusion_feats = self.fusion_conv(fusion_feats)

        return fusion_feats





