import logging
from typing import Sequence
import torch
from torchvision import utils
from enum import Enum
import math
from functools import partial
import os
import zarr

logger = logging.getLogger(__name__)

def get_next_sample(existing: Sequence[str], digits=None):
    if len(existing) < 1:
        next_sample = 0
        if digits is None:
            digits = 5
            logger.info(f"Number of digits not specified and no strings given to derive from. Defaulting to {digits}.")
    else:
        next_sample = max(int(s) for s in list(existing)) + 1
        if digits is None:
            digits = len(list(existing)[0])
        elif digits != len(list(existing)[0]):
            msg = f"Specified to use {digits} digits but found string with {len(list(existing)[0])} digits"
            raise ValueError(msg)
    next_sample_str = "{num:0{digits}d}".format(num=next_sample, digits=digits)
    return next_sample_str

def adjust_range(img: torch.Tensor, values=(0, 255)) -> torch.Tensor:
    return img.mul(values[1]).clamp_(values[0], values[1])


def to_numpy(img: torch.Tensor) -> np.array:
    return img.numpy()


def to_cpu(img: torch.Tensor) -> torch.Tensor:
    return img.to("cpu")


def to_dtype(img: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
    return img.to(dtype)


def griddify(img: torch.Tensor) -> torch.Tensor:
    num_samples = img.shape[0]
    samples_per_row = int(math.sqrt(num_samples))
    img = utils.make_grid(img, samples_per_row)
    return img


class PreProcessOptions(Enum):
    TO_UINT8 = partial(to_dtype, dtype=torch.uint8)
    ADJUST_RANGE_0_255 = partial(adjust_range, values=(0, 255))
    TO_NUMPY = partial(to_numpy)
    TO_CPU = partial(to_cpu)
    GRIDDIFY = partial(griddify)

    def __call__(self, *args):
        return self.value(*args)



class InferenceSaver: # todo: rename
    def __init__(self, channel_assignment, sample_digits=5):
        self.sample_digits = sample_digits
        self.channel_assignment = channel_assignment

    def save_sample(self, path, samples, timesteps: Sequence[int | str]):
        num_samples = samples.shape[0]
        samples_per_row = int(math.sqrt(num_samples))
        if samples.ndim == 5 and len(timesteps) != samples.shape[1]:
            raise ValueError(
                f"Number of timesteps ({len(timesteps)}) doesn't match length of time dimension ({samples.shape[1]})"
            )
        zarr_path = os.path.join(path, "samples", f"grid_{num_samples}.zarr")
        zarr_grp = zarr.group(store=zarr.DirectoryStore(zarr_path))
        next_sample = get_next_sample(zarr_grp.keys(), digits=self.sample_digits)
        sample_grp = zarr_grp.require_group(next_sample)
        for k, t in enumerate(timesteps):
            time_grp = sample_grp.require_group(t)
            if samples.ndim == 5:
                time_sample = samples[:, k, ...]
            else:
                time_sample = samples
            for img_name, (channel_slice, preprocessfuncs) in self.channel_assignment.items():
                img_data = time_sample[:, slice(*channel_slice), ...]
                for func_option in preprocessfuncs:
                    if func_option is not None:
                        img_data = func_option(img_data)
                time_grp.create_dataset(img_name, data=img_data)
