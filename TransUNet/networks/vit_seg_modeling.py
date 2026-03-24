# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size # hidden_size also num_heads

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, sequence_length, hidden_size) = (batch_size, 196, 768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights, None


def _repeat_kv(x, n_rep):
    """Expand KV head groups along the head dimension (same layout as multihead_diffattn.repeat_kv)."""
    bs, n_kv_groups, slen, width = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_groups, n_rep, slen, width)
        .reshape(bs, n_kv_groups * n_rep, slen, width)
    )


def _lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


try:
    from apex.normalization import FusedRMSNorm as _DiffRMSNorm
except ModuleNotFoundError:
    try:
        from rms_norm import RMSNorm as _DiffRMSNorm
    except ModuleNotFoundError:

        class _DiffRMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-5, elementwise_affine=True):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

            def forward(self, x):
                x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                if self.weight is not None:
                    x = x * self.weight
                return x


class DiffAttention(nn.Module):
    """Differential attention (see multihead_diffattn.MultiheadDiffAttn), ViT-style API (no RoPE / no causal mask)."""

    def __init__(self, config, vis, depth=0):
        super(DiffAttention, self).__init__()
        self.vis = vis
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.transformer["num_heads"]
        num_kv_heads = config.transformer.get("num_kv_heads", None)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_attention_heads
        self.n_rep = self.num_attention_heads // self.num_kv_heads

        self.head_dim = self.hidden_size // self.num_attention_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.query = Linear(self.hidden_size, self.hidden_size)
        self.key = Linear(self.hidden_size, self.hidden_size // self.n_rep)
        self.value = Linear(self.hidden_size, self.hidden_size // self.n_rep)
        self.out = Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.lambda_init = _lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = _DiffRMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, hidden_states, return_gamma=False):
        bsz, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = q.view(bsz, seq_len, 2 * self.num_attention_heads, self.head_dim)
        k = k.view(bsz, seq_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 2 * self.head_dim)

        q = q.transpose(1, 2)
        k = _repeat_kv(k.transpose(1, 2), self.n_rep)
        v = _repeat_kv(v.transpose(1, 2), self.n_rep)
        q = q * self.scaling

        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = Softmax(dim=-1)(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_split = attn_weights.view(bsz, self.num_attention_heads, 2, seq_len, seq_len)
        weights = (
            (attn_split[:, :, 0, :, :] - lambda_full * attn_split[:, :, 1, :, :])
            if self.vis
            else None
        )
        gamma = None
        if self.vis or return_gamma:
            a1 = attn_split[:, :, 0, :, :].mean(dim=1)
            a2 = attn_split[:, :, 1, :, :].mean(dim=1)
            m1 = a1.mean(dim=1)
            m2 = a2.mean(dim=1)
            side = int(round(seq_len ** 0.5))
            if side * side == seq_len:
                m1 = m1.view(bsz, side, side)
                m2 = m2.view(bsz, side, side)
                gamma = 1.0 - (m1 - m2).abs().mean(dim=(1, 2))
            else:
                gamma = 1.0 - (m1 - m2).abs().mean(dim=1)
            gamma = gamma.clamp(0.0, 1.0)
        attn_weights = self.attn_dropout(attn_weights)
        attn_weights = attn_weights.view(bsz, self.num_attention_heads, 2, seq_len, seq_len)
        attn_weights = attn_weights[:, :, 0, :, :] - lambda_full * attn_weights[:, :, 1, :, :]

        context_layer = torch.matmul(attn_weights, v)
        context_layer = self.subln(context_layer)
        context_layer = context_layer * (1 - self.lambda_init)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(bsz, seq_len, self.num_attention_heads * 2 * self.head_dim)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights, gamma


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        # img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x
        embeddings = self.dropout(embeddings)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis, layer_idx=0):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        use_diff = bool(config.transformer.get("use_diff_attn", False))
        if use_diff:
            self.attn = DiffAttention(config, vis, depth=layer_idx)
        else:
            self.attn = Attention(config, vis)

    def forward(self, x, return_attn_gamma=False):
        h = x
        x = self.attention_norm(x)
        if isinstance(self.attn, DiffAttention):
            x, weights, gamma = self.attn(x, return_gamma=return_attn_gamma)
        else:
            x, weights, gamma = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights, gamma

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.collect_diff_gamma = False
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            self.layer.append(Block(config, vis, layer_idx=i))

    def forward(self, hidden_states):
        attn_weights = []
        last_gamma = None
        n_layers = len(self.layer)
        for i, layer_block in enumerate(self.layer):
            ra = self.collect_diff_gamma and (i == n_layers - 1)
            hidden_states, weights, gamma = layer_block(hidden_states, return_attn_gamma=ra)
            if self.vis and weights is not None:
                attn_weights.append(weights)
            if gamma is not None:
                last_gamma = gamma
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights, last_gamma


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        print(last_gamma)
        encoded, attn_weights, last_gamma = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features, last_gamma


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        # self.conv_more = Conv2dReLU(
        #     config.hidden_size,
        #     1024,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=True,
        # )
        # self.conv_more_ = Conv2dReLU(
        #     1024,
        #     2048,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     use_batchnorm=True,
        # )
        self.fc= nn.Linear(config.hidden_size,2048)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.avgpool= nn.AdaptiveAvgPool1d(1)
        # self.transfermer_f34=nn.Transformer(nhead=4, num_encoder_layers=3,d_model=768)
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(features[0].shape[2]/2), int(features[0].shape[3]/2)
        ### unified multi-scale transformer

        x = hidden_states.permute(0, 2, 1)
        x = self.avgpool(x)
        ### for vis
        # vis = x.contiguous().view(B, hidden, h, w)
        # vis = functional.interpolate(vis, size=(224,224), mode="bilinear", align_corners=False)

        x = x.contiguous().view(B, hidden)
        x = self.fc(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)

        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x0, attn_weights, features, last_gamma = self.transformer(x)  # (B, n_patch, hidden)
        x1 = self.decoder(x0, features)
        f=list(reversed(features))
        f.append(x1)
        f.insert(0, x)
        return f, x1, last_gamma

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            #
            # posemb_new = self.transformer.embeddings.position_embeddings
            # if posemb.size() == posemb_new.size():
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # elif posemb.size()[1]-1 == posemb_new.size()[1]:
            #     posemb = posemb[:, 1:]
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # else:
            #     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
            #     ntok_new = posemb_new.size(1)
            #     if self.classifier == "seg":
            #         _, posemb_grid = posemb[:, :1], posemb[0, 1:]
            #     gs_old = int(np.sqrt(len(posemb_grid)))
            #     gs_new = int(np.sqrt(ntok_new))
            #     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            #     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            #     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            #     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
            #     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            #     posemb = posemb_grid
            #
            #     # self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            #     self.transformer.embeddings.position_embeddings[:,0:posemb.shape[1],:]=np2th(posemb)
            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


