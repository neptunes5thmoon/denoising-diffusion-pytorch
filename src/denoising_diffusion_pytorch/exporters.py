import logging
import math
import os
from enum import Enum
from functools import partial
from typing import Sequence

import torch
import zarr
from PIL import Image
from torchvision import utils

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


def adjust_range(img: torch.Tensor, *args, values=(0, 255)) -> torch.Tensor:
    return img.mul(values[1]).clamp_(values[0], values[1])


def to_numpy(img: torch.Tensor, *args) -> np.array:
    return img.numpy()


def to_cpu(img: torch.Tensor, *args) -> torch.Tensor:
    return img.to("cpu")


def to_dtype(img: torch.Tensor, *args, dtype=torch.uint8) -> torch.Tensor:
    return img.to(dtype)

def griddify(img: torch.Tensor, *args) -> torch.Tensor:
    num_samples = img.shape[0]
    samples_per_row = int(math.sqrt(num_samples))
    img = utils.make_grid(img, samples_per_row)
    return img


class PostProcessOptions(Enum):
    TO_UINT8 = partial(to_dtype, dtype=torch.uint8)
    ADJUST_RANGE_0_255 = partial(adjust_range, values=(0, 255))
    TO_NUMPY = partial(to_numpy)
    TO_CPU = partial(to_cpu)
    GRIDDIFY = partial(griddify)

    def __call__(self, *args):
        return self.value(*args)


class SampleExporter(object):
    def __init__(self, channel_assignment, sample_digits=5, file_format=".zarr", sample_batch_size=1):
        self.sample_digits = sample_digits
        self.channel_assignment = channel_assignment
        self.file_format = file_format
        self.sample_batch_size = sample_batch_size

    def _make_dir_zarr(self, path):
        zarr_grp = zarr.group(store=zarr.DirectoryStore(path))
        next_sample = get_next_sample(zarr_grp.keys(), digits=self.sample_digits)
        sample_grp = zarr_grp.require_group(next_sample)
        return sample_grp

    def _save_img_zarr(self, grp, name, data):
        grp.create_dataset(name, data=data)

    def _make_dir_png(self, path):
        os.makedirs(path, exist_ok=True)
        next_sample = get_next_sample(os.listdir(path), digits=self.sample_digits)
        sample_dir = os.path.join(path, next_sample)
        os.makedirs(sample_dir)
        return sample_dir

    def _save_img_png(self, path, name, data):
        img = Image.fromarray(data.permute(1, 2, 0))
        fp = os.path.join(path, f"{name}.png")
        img.save(fp, format="PNG")

    def save_sample(self, path, samples):
        for batch_start in range(0, samples.shape[0] - self.sample_batch_size + 1, self.sample_batch_size):
            sample = samples[batch_start : batch_start + self.sample_batch_size]
            if self.file_format == ".zarr":
                sample_path = self._make_dir_zarr(path)
            elif self.file_format == ".png":
                sample_path = self._make_dir_png(path)
            else:
                msg = f"Unknown file format ({self.file_format}) requested."
                raise ValueError(msg)
            for img_name, (channel_slice, preprocessfuncs) in self.channel_assignment.items():
                img_data = sample[:, slice(*channel_slice), ...]
                for func_option in preprocessfuncs:
                    if func_option is not None:
                        img_data = func_option(img_data, img_name)
                if self.file_format == ".zarr":
                    self._save_img_zarr(sample_path, img_name, img_data)
                elif self.file_format == ".png":
                    self._save_img_png(sample_path, img_name, img_data)
                else:
                    msg = f"Unknown file format ({self.file_format}) requested."
                    raise ValueError(msg)
