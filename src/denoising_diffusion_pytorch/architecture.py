import math
from functools import partial

import torch
import torch.nn.functional as f
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.convenience import (
    cast_tuple,
    default,
    divisible_by,
    exists,
)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
        )

    def forward(self, *args):
        return self.up(*args)


class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.down = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1),
        )

    def forward(self, *args):
        return self.down(*args)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return f.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return f.normalize(x, dim=1) * self.g * self.scale


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        if not divisible_by(dim, 2):
            msg = f"`dim` in {self.__class__.__name__} needs to be divisible by 2, but got {dim}"
            raise ValueError(msg)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        classes_emb_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        emb_dim = 0
        if time_emb_dim is not None:
            emb_dim += int(time_emb_dim)
        if classes_emb_dim is not None:
            emb_dim += int(classes_emb_dim)
        if emb_dim > 0:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
        else:
            self.mlp = None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, class_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv)

        mk, mv = (repeat(t, "h c n -> b h c n", b=b) for t in self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()

        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads) for t in qkv)

        mk, mv = (repeat(t, "h n d -> b h n d", b=b) for t in self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes=None,
        cond_drop_prob=0.5,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,  # defaults to full attention only for inner most layer
        flash_attn=False,
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob
        self.num_classes = num_classes

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        if len(full_attn) != len(dim_mults):
            msg = (
                f"`dim_mults` and `full_attn` need to be of same length but got {dim_mults} of length "
                f"{len(dim_mults)} and {full_attn} of length {len(full_attn)}"
            )
            raise ValueError(msg)

        # class embeddings
        if self.num_classes is not None:
            self.classes_emb = nn.Embedding(num_classes, dim)
            self.null_classes_emb = nn.Parameter(torch.randn(dim))
            classes_dim = dim * 4
            self.classes_mlp = nn.Sequential(
                nn.Linear(dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim),
            )
        else:
            classes_dim = None
        # prepare blocks

        full_attention_block = partial(Attention, flash=flash_attn)
        resnet_block = partial(
            ResnetBlock,
            time_emb_dim=time_dim,
            classes_emb_dim=classes_dim,
            dropout=dropout,
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = full_attention_block if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(
                            dim_in,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        (Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = full_attention_block(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = full_attention_block if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        (Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward_with_cond_scale(
        self,
        x,
        time,
        x_self_cond=None,
        classes=None,
        cond_scale=1.0,
        rescaled_phi=0.0,
    ):
        logits = self.forward(
            x,
            time,
            x_self_cond=x_self_cond,
            classes=classes,
            cond_drop_prob=0.0,
        )
        if cond_scale == 1:
            return logits
        null_logits = self.forward(
            x,
            time,
            x_self_cond=x_self_cond,
            classes=classes,
            cond_drop_prob=1.0,
        )
        scaled_logits = null_logits + (logits - null_logits) * cond_scale
        if rescaled_phi == 0.0:
            return scaled_logits, null_logits
        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)
        return interpolated_rescaled_logits, null_logits

    def forward(
        self,
        x,
        time,
        x_self_cond=None,
        classes=None,
        cond_drop_prob=None,
    ):
        if not all(divisible_by(d, self.downsample_factor) for d in x.shape[-2:]):
            msg = "Your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}"
            raise ValueError(msg)
        batch = x.shape[0]
        if self.num_classes is not None:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
            classes_emb = self.classes_emb(classes)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=x.device)
                null_classes_emb = repeat(self.null_classes_emb, "d -> b d", b=batch)
                classes_emb = torch.where(
                    rearrange(keep_mask, "b -> b 1"),
                    classes_emb,
                    null_classes_emb,
                )
            c = self.classes_mlp(classes_emb)
        else:
            c = None

        # self conditioning
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        # unet
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, c)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
