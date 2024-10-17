import logging
import math
import os
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import mlflow
import numpy as np
import torch
import torch.nn.functional as f
from accelerate import Accelerator
from einops import rearrange, reduce
from ema_pytorch import EMA
from more_itertools import batched
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.convenience import (
    cycle,
    default,
    divisible_by,
    exists,
    has_int_squareroot,
    identity,
    num_to_groups,
)
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from denoising_diffusion_pytorch.version import __version__

logger = logging.getLogger(__name__)
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def gaussian_weights(width, height, var=0.01):
    """Generates a gasian mask of weights for tile contributions"""
    midpoint = (width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (width * width) / (2 * var))
        / np.sqrt(2 * np.pi * var)
        for x in range(width)
    ]
    midpoint = height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (height * height) / (2 * var))
        / np.sqrt(2 * np.pi * var)
        for y in range(height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=None,
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
        use_cfg_plus_plus=False,  # https://arxiv.org/pdf/2406.08070
        immiscible=False,
        channel_weights=None,
    ):
        super().__init__()
        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective
        if objective not in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }:
            msg = (
                f"""`objective` must be either `"pred_noise"` (predict noise) or `"pred_x0"` (predict image start)"""
                f""" or `"pred_v"` (predict v [v-parameterization as defined in appendix D of progressive"""
                f""" distillation paper, used in imagen-video successfully]), but got {objective}"""
            )
            raise ValueError(msg)

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            msg = f"""Unkown `beta_schedule`: {beta_schedule}. Options are `"linear"`, `"cosine"` or `"sigmoid"`."""
            raise ValueError(msg)

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = f.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        # use cfg++ when ddim sampling
        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training
        if self.sampling_timesteps > timesteps:
            msg = (
                f"`sampling_timesteps` ({self.sampling_timesteps}) needs to be smaller "
                f"or equal to `timesteps` ({timesteps})"
            )
            raise ValueError(msg)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.to(torch.float32)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(torch.float32)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).to(torch.float32),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - alphas_cumprod).to(torch.float32),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod).to(torch.float32),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1).to(torch.float32),
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer("posterior_variance", posterior_variance.to(torch.float32))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)).to(torch.float32),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(
                torch.float32
            ),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (
                (1.0 - alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - alphas_cumprod)
            ).to(torch.float32),
        )
        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            self.register_buffer(
                "loss_weight", (maybe_clipped_snr / snr).to(torch.float32)
            )
        elif objective == "pred_x0":
            self.register_buffer("loss_weight", maybe_clipped_snr.to(torch.float32))
        elif objective == "pred_v":
            self.register_buffer(
                "loss_weight", (maybe_clipped_snr / (snr + 1)).to(torch.float32)
            )

        if channel_weights is not None:
            if model.channels != len(channel_weights):
                msg = (
                    f"if `channel_weights` are given the length must match the number of channels ({model.channels}),"
                    f" but got {channel_weights} of length {len(channel_weights)}"
                )
                raise ValueError(msg)
            channel_weights_t = torch.nn.functional.normalize(
                torch.FloatTensor(channel_weights), p=1, dim=0
            )
            self.register_buffer("channel_weights", channel_weights_t.to(torch.float32))
        else:
            self.channel_weights = None

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    def model_predictions(
        self,
        x,
        t,
        x_self_cond=None,
        classes=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        if classes is None:
            model_output = self.model(x, t, x_self_cond)
        else:
            model_output, model_output_null = self.model.forward_with_cond_scale(
                x,
                t,
                x_self_cond=x_self_cond,
                classes=classes,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
            )

        if self.objective == "pred_noise":
            if self.use_cfg_plus_plus and classes is not None:
                pred_noise = model_output_null
            else:
                pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            if self.use_cfg_plus_plus and classes is not None:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                pred_noise = self.predict_noise_from_start(
                    x, t, maybe_clip(model_output_null)
                )

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            if self.use_cfg_plus_plus and classes is not None:
                x_start_for_pred_noise = self.predict_start_from_v(
                    x, t, model_output_null
                )
                pred_noise = self.predict_noise_from_start(
                    x, t, maybe_clip(x_start_for_pred_noise)
                )
            else:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x,
        t,
        x_self_cond=None,
        classes=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        clip_denoised=True,
    ):
        preds = self.model_predictions(
            x,
            t,
            x_self_cond=x_self_cond,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(
        self,
        x,
        t: int,
        x_self_cond=None,
        classes=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        noise=None,
        clip_denoised=True,
    ):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            x_self_cond=x_self_cond,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        if noise is None:
            noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        elif t == 0:
            noise = 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        if self.self_condition is None:
            x_start = None
        return pred_img, x_start

    @torch.inference_mode()
    def blockwise_sample(
        self,
        data,
        noise,
        self_cond,
        block_weight,
        blockshape,
        sampling_func,
        t,
        batchsize=1,
        **sample_fn_kwargs,
    ):
        if np.any(np.array(data.shape) % np.array(blockshape)):
            msg = f"For now, the array shape ({data.shape}) needs to be multiple of the blockshape ({blockshape})."
            raise ValueError(msg)
        splits = np.array(data.shape) / np.array(blockshape)

        # Split into blocks
        block_list = [data]
        noise_list = [noise]
        if self_cond is not None:
            self_cond_list = [self_cond]

        # loop over each axis to generate split along that axis
        for axis, i in enumerate(blockshape):
            new_block_list = []
            new_noise_list = []
            for el in block_list:
                new_block_list.extend(torch.split(el, i, axis))
            block_list = new_block_list
            for el in noise_list:
                new_noise_list.extend(torch.split(el, i, axis))
            noise_list = new_noise_list
            if self_cond is not None:
                new_self_cond_list = []
                for el in self_cond_list:
                    new_self_cond_list.extend(torch.split(el, i, axis))
                self_cond_list = new_self_cond_list
        if self_cond is None:
            self_cond_list = [
                None,
            ] * len(noise_list)

        # Process blocks in batches
        processed_blocks = []
        processed_xstarts = []
        for block_batch, noise_batch, cond_batch in zip(
            batched(block_list, batchsize),
            batched(noise_list, batchsize),
            batched(self_cond_list, batchsize),
        ):
            # batch blocks by concatenating along batch dimension
            batch_data = torch.cat(block_batch, axis=0)
            batch_noise = torch.cat(noise_batch, axis=0)
            if self.self_condition:
                batch_cond = torch.cat(cond_batch, axis=0)
            else:
                batch_cond = None
            # process each block
            processed_batch, x_start = sampling_func(
                batch_data,
                t=t,
                x_self_cond=batch_cond,
                noise=batch_noise,
                **sample_fn_kwargs,
            )
            # weight each block
            processed_batch = processed_batch * block_weight
            # turn back into a list by splitting along batch dimension
            processed_blocks.extend(torch.split(processed_batch, 1, 0))
            # if self conditioning, do same for x, otherwise we don't need to keep track of this
            if self.self_condition:
                x_start = x_start * block_weight
                processed_xstarts.extend(torch.split(x_start, 1, 0))
            else:
                x_start = None

        # Reassemble into array
        assembly_list = processed_blocks
        # Glue back together what was previously split
        for axis, i in zip(reversed(range(len(splits))), reversed(splits)):
            new_assembly_list = []
            for assembly in batched(assembly_list, int(i)):
                new_assembly_list.append(torch.cat(assembly, axis))
            assembly_list = new_assembly_list
        if len(assembly_list) != 1:
            msg = (
                f"`assembly_list` should be a single element after for loop but got {assembly_list} of length"
                f" {len(assembly_list)}"
            )
            raise RuntimeError(msg)

        assembly_list = assembly_list[0]
        if self.self_condition:
            x_assembly_list = processed_xstarts
            for axis, i in zip(reversed(range(len(splits))), reversed(splits)):
                new_assembly_list = []
                for assembly in batched(x_assembly_list, int(i)):
                    new_assembly_list.append(torch.cat(assembly, axis))
                x_assembly_list = new_assembly_list
            if len(x_assembly_list) != 1:
                msg = (
                    f"`x_assembly_list` should be a single element after for loop but got {x_assembly_list} of length"
                    f" {len(x_assembly_list)}"
                )
                raise RuntimeError(msg)
            x_start = x_assembly_list[0]
        return assembly_list, x_start

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape,  # (bs, ch, x, y)
        classes=None,
        return_all_timesteps=False,
        blockshape=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        var=0.1,
        n_batch_blocks=10,
    ):
        batch, device = shape[0], self.device

        # Initialize weighting of chunks if inference should be run blockwise
        if blockshape is not None:
            if blockshape[2] != blockshape[3]:
                msg = "Only square block shapes are supported"
                raise ValueError(msg)
            if blockshape[2] % 2 != 0:
                msg = "Only even block shapes are supported"
                raise ValueError(msg)
            if batch != 1:
                n_batch_blocks = 1
            overlap = int(blockshape[2] / 2)
            chunk_weight = gaussian_weights(blockshape[2], blockshape[3], var=var)
            chunk_weight_norm = chunk_weight + np.roll(
                chunk_weight, (overlap, overlap), (0, 1)
            )
            chunk_weight = chunk_weight / chunk_weight_norm
            chunk_weight = torch.Tensor(chunk_weight).to(device)

        # Initialize image as pure standard gaussian noise
        img = torch.randn(shape, device=self.device)

        # initalize list for intermediate timesteps
        if return_all_timesteps:
            imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            if blockshape is None:  # Single-shot prediction
                self_cond = x_start if self.self_condition else None
                img, x_start = self.p_sample(
                    img,
                    t,
                    x_self_cond=self_cond,
                    classes=classes,
                    rescaled_phi=rescaled_phi,
                    cond_scale=cond_scale,
                )
            else:
                noise = torch.randn(shape, device=self.device)  # initalize noise
                img1, x_start1 = self.blockwise_sample(
                    img,
                    noise,
                    x_start,
                    chunk_weight,
                    blockshape,
                    self.p_sample,
                    t,
                    batchsize=n_batch_blocks,
                    classes=classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )  # run inference in blocks
                img2, x_start2 = self.blockwise_sample(
                    torch.roll(img, (overlap, overlap), (2, 3)),
                    torch.roll(noise, (overlap, overlap), (2, 3)),
                    (
                        torch.roll(x_start, (overlap, overlap), (2, 3))
                        if x_start is not None
                        else None
                    ),
                    chunk_weight,
                    blockshape,
                    self.p_sample,
                    t,
                    batchsize=n_batch_blocks,
                    classes=classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )  # run inference in blocks, but offset by overlap in x,y
                img2 = torch.roll(
                    img2, (-overlap, -overlap), (2, 3)
                )  # undo offset by overlap in x, y
                if self.self_condition:  # only need this if self conditioning is on
                    x_start = x_start1 + x_start2
                img = img1 + img2  # smooth out prediction across blocks
            if return_all_timesteps:
                imgs.append(img)
        if return_all_timesteps:
            ret = torch.stack(imgs, dim=1)
        else:
            ret = img

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(
        self,
        x,
        t: tuple[int, int],
        x_self_cond=None,
        classes=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        noise=None,
        clip_denoised=True,
    ):
        time, time_next = t
        batch = x.shape[0]
        time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)

        pred_noise, x_start, *_ = self.model_predictions(
            x,
            time_cond,
            x_self_cond,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_x_start=clip_denoised,
            rederive_pred_noise=True,
        )
        if time_next < 0:
            img = x_start
        else:
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                self.ddim_sampling_eta
                * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            if noise is None:
                noise = torch.randn_like(x)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return img, x_start

    @torch.inference_mode()
    def ddim_sample_loop(
        self,
        shape,
        classes=None,
        return_all_timesteps=False,
        blockshape=None,
        cond_scale=6.0,
        rescaled_phi=0.7,
        var=0.1,
        n_batch_blocks=10,
    ):
        batch = shape[0]

        if blockshape is not None:
            if blockshape[2] != blockshape[3]:
                msg = "Only square block shapes are supported"
                raise ValueError(msg)
            if blockshape[2] % 2 != 0:
                msg = "Only even block shapes are supported"
                raise ValueError(msg)
            if batch != 1:
                n_batch_blocks = 1
            overlap = int(blockshape[2] / 2)
            chunk_weight = gaussian_weights(blockshape[2], blockshape[3], var=var)
            chunk_weight_norm = chunk_weight + np.roll(
                chunk_weight, (overlap, overlap), (0, 1)
            )
            chunk_weight = chunk_weight / chunk_weight_norm
            chunk_weight = torch.Tensor(chunk_weight).to(self.device)

        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        if return_all_timesteps:
            imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            if blockshape is None:
                x_self_cond = x_start if self.self_condition else None
                img, x_start = self.ddim_sample(
                    img,
                    (time, time_next),
                    x_self_cond=x_self_cond,
                    classes=classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )

            else:
                noise = torch.randn(shape, device=self.device)
                img1, x_start1 = self.blockwise_sample(
                    img,
                    noise,
                    x_start,
                    chunk_weight,
                    blockshape,
                    self.ddim_sample,
                    (time, time_next),
                    batchsize=n_batch_blocks,
                    classes=classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )
                img2, x_start2 = self.blockwise_sample(
                    torch.roll(img, (overlap, overlap), (2, 3)),
                    torch.roll(noise, (overlap, overlap), (2, 3)),
                    torch.roll(x_start) if x_start is not None else None,
                    chunk_weight,
                    blockshape,
                    self.ddim_sample,
                    (time, time_next),
                    batchsize=n_batch_blocks,
                    classes=classes,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )
                img2 = torch.roll(img2, (-overlap, -overlap), (2, 3))
                if self.self_condition:
                    x_start = x_start1 + x_start2
                img = img1 + img2
            if return_all_timesteps:
                imgs.append(img)
        if return_all_timesteps:
            ret = torch.stack(imgs, dim=1)
        else:
            ret = img

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False, **kwargs):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample_loop
        )
        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            **kwargs,
        )

    @torch.inference_mode()
    def interpolate(
        self, x1, x2, classes, t=None, lam=0.5, cond_scale=6.0, rescaled_phi=0.7
    ):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        if x1.shape != x2.shape:
            msg = f"Shapes of images to interpolate between needs to match but got {x1.shape} and {x2.shape}"
            raise ValueError(msg)

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = (self.q_sample(x, t=t_batched) for x in (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(
            reversed(range(0, t)),
            desc="interpolation sample time step",
            total=t,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                img,
                i,
                self_cond,
                classes,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
            )

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(
            rearrange(t, "b ... -> b (...)") for t in (x_start, noise)
        )
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self, x_start, t, classes=None, noise=None, offset_noise_strength=None
    ):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:  # noqa: PLR2004,S311
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond=x_self_cond, classes=classes)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            msg = (
                f"""`objective` must be either `"pred_noise"` (predict noise) or `"pred_x0"` (predict image start)"""
                f""" or `"pred_v"` (predict v [v-parameterization as defined in appendix D of progressive"""
                f""" distillation paper, used in imagen-video successfully]), but got {self.objective}"""
            )
            raise ValueError(msg)

        loss = f.mse_loss(model_out, target, reduction="none")

        # apply channel weights
        if self.channel_weights is not None:
            loss = loss * self.channel_weights.view(1, c, 1, 1)
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        (
            b,
            _,
            h,
            w,
            device,
            img_size,
        ) = (*img.shape, img.device, self.image_size)
        if h != img_size or w != img_size:
            msg = f"Height and width of image must be {img_size}, but got {h} and {w}"
            raise ValueError(msg)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


# trainer class


class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        exporter,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.0,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        dataloader_nworkers=None,
        persistent_workers=True,
        prefetch_factor=2,
        shuffle_dataloader=True,
        repeat_data=True,
    ):
        super().__init__()
        if dataloader_nworkers is None:
            dataloader_nworkers = cpu_count()
        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )
        # saver
        self.exporter = exporter

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters
        if not has_int_squareroot(num_samples):
            msg = f"Number of samples must have an integer square root but got {num_samples=}"
            raise ValueError(msg)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        if train_batch_size * gradient_accumulate_every < 16:  # noqa: PLR2004
            msg = "Your effective batch size (`train_batch_size` x `gradient_accumulate_every`) should be at least 16"
            raise UserWarning(msg)

        self.train_num_steps = train_num_steps
        if not repeat_data and self.train_num_steps > len(dataset):
            self.train_num_steps = len(dataset)
        self.image_size = diffusion_model.image_size
        self.sampling_timesteps = diffusion_model.sampling_timesteps
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = dataset
        if len(self.ds) < 100:  # noqa: PLR2004
            msg = (
                f"You should have at least 100 images for training, at least 10k images recommended. "
                f"Found {len(self.ds)}"
            )
            raise UserWarning(msg)
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=shuffle_dataloader,
            pin_memory=True,
            num_workers=dataloader_nworkers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.val_dl = None

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model, beta=ema_decay, update_every=ema_update_every
            )
            self.ema.to(self.device)
            self.sampler = self.ema.ema_model

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                msg = (
                    "WARNING: Robust FID computation requires a lot of generated samples "
                    "and can therefore be very time consuming."
                    "Consider using DDIM sampling to save time."
                )
                logger.warning(msg, main_process_only=True)

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx,
            )

        if save_best_and_latest_only and not calculate_fid:
            msg = (
                "`calculate_fid` must be True to provide a means for model evaluation for "
                "`save_best_and_latest_only`."
            )
            raise ValueError(msg)
        if save_best_and_latest_only:
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        total_milestones = math.ceil(self.train_num_steps / self.save_and_sample_every)
        self.milestone_digits = len(str(total_milestones))

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            "version": __version__,
        }

        ckpt_dir = self.results_folder / f"ckpt_{milestone:0{self.milestone_digits}d}"
        ckpt_dir.mkdir(exist_ok=True)
        model_path = str(ckpt_dir / f"model_{milestone:0{self.milestone_digits}d}.pt")
        torch.save(data, model_path)

    def load_last(self):
        milestones = [
            int(ckpt.split("_")[1]) for ckpt in os.listdir(self.results_folder)
        ]
        if len(milestones) > 0:
            self.load(max(milestones))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(
                self.results_folder
                / f"ckpt_{milestone:0{self.milestone_digits}d}"
                / f"model_{milestone:0{self.milestone_digits}d}.pt"
            ),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            logger.info(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.load_last()

        if self.val_dl is not None:
            for data, target in self.val_dl:
                all_data = accelerator.gather(data)
                all_target = accelerator.gather(target)
                self.loader_exporter.save_sample(
                    str(self.results_folder / "reference"),
                    torch.cat([all_data, all_target], dim=1),
                )
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = [d.to(device) for d in next(self.dl)]

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)
                mlflow.log_metric("loss", total_loss, step=self.step)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(
                        self.step, self.save_and_sample_every
                    ):
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                        with torch.inference_mode():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = [
                                self.ema.ema_model.sample(batch_size=n) for n in batches
                            ]

                        all_images = torch.cat(all_images_list, dim=0)
                        if self.channels <= 3:  # noqa: PLR2004
                            utils.save_image(
                                all_images,
                                str(
                                    self.results_folder
                                    / f"sample_{milestone:0{self.milestone_digits}d}.png"
                                ),
                                nrow=int(math.sqrt(self.num_samples)),
                            )
                        else:
                            self.exporter.save_sample(
                                str(
                                    self.results_folder
                                    / f"ckpt_{milestone:0{self.milestone_digits}d}"
                                ),
                                all_images,
                            )

                        # whether to calculate fid
                        if self.calculate_fid:
                            if self.channels <= 3:  # noqa: PLR2004
                                fid_score = self.fid_scorer.fid_score()
                                logger.info(
                                    f"fid_score: {fid_score}", main_process_only=True
                                )

                            else:
                                msg = "FID score cannot be calculated for data with more than 3 channels."
                                raise ValueError(msg)

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)
        logger.info("Training complete", main_process_only=True)


if __name__ == "__main__":
    from architecture import Unet

    logger.info("With classes")
    num_classes = 10
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_classes=num_classes,
        cond_drop_prob=1,
        channels=6,
    )
    diffusion = GaussianDiffusion(
        model, image_size=128, timesteps=1000, sampling_timesteps=250
    ).cuda()
    training_images = torch.randn((8, 6, 128, 128)).cuda()
    image_classes = torch.randint(0, num_classes, (8,)).cuda()
    loss = diffusion(training_images, image_classes)
    loss.backward()
    sample_classes = torch.randint(0, num_classes, (20,)).cuda()
    sampled_images = diffusion.sample(20, classes=sample_classes)
    logger.debug(f"{sampled_images.shape=}")
    sampled_images = diffusion.p_sample_loop((20, 6, 128, 128), sample_classes)
    sampled_images = diffusion.ddim_sample_loop((20, 6, 128, 128), sample_classes)

    # Without classes
    logger.info("Without classes")
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), cond_drop_prob=0.5, channels=6)
    diffusion = GaussianDiffusion(
        model, image_size=128, timesteps=1000, sampling_timesteps=250
    ).cuda()
    training_images = torch.randn((8, 6, 128, 128)).cuda()
    loss = diffusion(training_images)
    loss.backward()
    sampled_images = diffusion.sample(20)
    logger.debug(f"{sampled_images.shape=}")
    sampled_images = diffusion.p_sample_loop((20, 6, 128, 128))
    sampled_images = diffusion.ddim_sample_loop((20, 6, 128, 128))
