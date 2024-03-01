import logging
import math
import os
from enum import Enum
from functools import partial
from typing import Sequence, Union, Tuple, Optional, Dict, Literal
import numpy as np
import torch
import zarr
from PIL import Image
from torchvision import utils
import distinctipy
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


def convert_color_to_float(
    color: Union[Tuple[int, int, int], Tuple[float, float, float]]
) -> Tuple[float, float, float]:
    if len(color) != 3:
        msg = f"Color tuple {color} is not of length 3"
        raise ValueError(msg)
    if not all(isinstance(col, type(color[0])) for col in color):
        msg = f"Inconsistent types in color tuple: {tuple(type(col) for col in color)}"
        raise ValueError(msg)
    if isinstance(color[0], int):
        for col in color:
            if col < 0 or col > 255:
                msg = f"Integer color values should be in range [0, 255]; found {color}"
                raise ValueError(msg)
        return tuple(col / 255.0 for col in color)
    elif isinstance(color[0], float):
        for col in color:
            if col < 0 or col > 1:
                msg = f"Float color values should be in range [0, 1]; found {color}"
                raise ValueError(msg)
        return color
    else:
        msg = f"Can't handle color values of type {color[0]}"
        raise TypeError(msg)


def adjust_range(img: torch.Tensor, range_in=(-1, 1), range_out=(0,255)) -> torch.Tensor:
    img = (img - range_in[0])/(range_in[1]-range_in[0]) * (range_out[1]- range_out[0]) + range_out[0]
    return img.clamp_(range_out[0], range_out[1])


def to_numpy(img: torch.Tensor) -> np.array:
    return img.numpy()


def to_cpu(img: torch.Tensor) -> torch.Tensor:
    return img.to("cpu")


def to_dtype(img: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
    return img.to(dtype)


def griddify(img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        to_np = True
    else:
        to_np = False
    num_samples = img.shape[0]
    samples_per_row = int(math.sqrt(num_samples))
    img = utils.make_grid(img, samples_per_row)
    if to_np:
        img = img.numpy()
    return img


def colorize(img: np.array, colors: Optional[Sequence[Tuple[float, float, float]]] = None, color_threshold=0):
    # img ch, x, y
    if img.ndim == 4:  # s, ch, x, y
        color_axis = 1
    elif img.ndim == 3:  # ch, x, y
        color_axis = 0
    else:
        msg = f"Can't handle arrays with {img.ndim} dimensions for colorizing"
        raise ValueError(msg)
    if colors is None or len(colors) < img.shape[color_axis]:
        new_colors = distinctipy.get_colors(img.shape[color_axis], colors=colors)
        if colors is None:
            colors = new_colors
        else:
            colors = colors.extend(new_colors)
    img[img <= color_threshold] = 0
    # find for each pixel which image has max value
    max_lbl_id_arr = np.argmax(img, axis=color_axis, keepdims=True)
    # keep track of what those actual values are
    max_lbl_val_arr = np.take_along_axis(img, max_lbl_id_arr, axis=color_axis)
    target_shape = list(img.shape)
    target_shape[color_axis] = 3
    rgb_image = np.zeros(target_shape)
    # normalizing_image = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.uint8)
    for lbl_id, color in zip(range(img.shape[color_axis]), colors):
        lbl_bin_arr = max_lbl_id_arr == lbl_id
        lbl_arr = max_lbl_val_arr * lbl_bin_arr
        rgb_img_tpl = tuple(lbl_arr * col for col in color)
        rgb_image += np.concatenate(rgb_img_tpl, axis=color_axis)
    if np.issubdtype(img.dtype, np.integer):
        rgb_image = np.round(rgb_image).astype(img.dtype)

    return rgb_image


class PostProcessOptions(Enum):
    TO_UINT8 = partial(to_dtype, dtype=torch.uint8)
    ADJUST_RANGE_0_1_TO_0_255 = partial(adjust_range, range_in=(0,1), range_out=(0, 255))
    ADJUST_RANGE_NEG1_1_TO_0_255 = partial(adjust_range, range_in=(-1,1), range_out=(0,255))
    TO_NUMPY = partial(to_numpy)
    TO_CPU = partial(to_cpu)
    GRIDDIFY = partial(griddify)
    COLORIZE = partial(colorize)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


PostProcessOptionsNames = Literal[tuple(e.name for e in PostProcessOptions)]


class SampleExporter(object):
    def __init__(
        self,
        channel_assignment: Dict[str, Tuple[Tuple[int, int, int], Sequence[Union[None, PostProcessOptionsNames]]]],
        sample_digits: int = 5,
        file_format: Literal[".zarr", ".png"] = ".zarr",
        sample_batch_size: int = 1,
        colors=None,
        color_threshold=0,
        dir="samples",
    ):
        self.sample_digits = sample_digits
        self.channel_assignment = channel_assignment
        self.file_format = file_format
        self.sample_batch_size = sample_batch_size
        self.dir_name = dir
        if colors is None:
            self.colors = colors
        else:
            self.colors = []
            for color in colors:
                self.colors.append(convert_color_to_float(color))
        self.color_threshold = color_threshold

    def _make_dir_zarr(self, path):
        zarr_grp = zarr.group(store=zarr.DirectoryStore(os.path.join(path, f"{self.dir_name}.zarr")))
        next_sample = get_next_sample(zarr_grp.keys(), digits=self.sample_digits)
        sample_grp = zarr_grp.require_group(next_sample)
        return sample_grp

    def _save_img_zarr(self, grp, name, data):
        grp.create_dataset(name, data=data)

    def _make_dir_png(self, path):
        sample_path = os.path.join(path, self.dir_name)
        os.makedirs(sample_path, exist_ok=True)
        next_sample = get_next_sample(os.listdir(sample_path), digits=self.sample_digits)
        sample_dir = os.path.join(sample_path, next_sample)
        os.makedirs(sample_dir)
        return sample_dir

    def _save_img_png(self, path, name, data):
        img = Image.fromarray(data.transpose(1, 2, 0))
        fp = os.path.join(path, f"{name}.png")
        img.save(fp, format="PNG")

    def save_sample(self, parent_path, samples) -> int:
        if samples.shape[0] < self.sample_batch_size:
            msg = f"Can't export sample with `sample_batch_size` ({self.sample_batch_size}) larger than number of samples ({samples.shape[0]})"
            raise ValueError(msg)
        for batch_start in range(0, samples.shape[0] - self.sample_batch_size + 1, self.sample_batch_size):
            sample = samples[batch_start : batch_start + self.sample_batch_size]
            if self.file_format == ".zarr":
                sample_path = self._make_dir_zarr(parent_path)
            elif self.file_format == ".png":
                sample_path = self._make_dir_png(parent_path)
            else:
                msg = f"Unknown file format ({self.file_format}) requested."
                raise ValueError(msg)
            for img_name, (channel_slice, preprocessfuncs) in self.channel_assignment.items():
                img_data = sample[:, slice(*channel_slice), ...]
                for func_option in preprocessfuncs:
                    if func_option is not None:
                        func_option = PostProcessOptions[func_option]
                        if func_option == PostProcessOptions.COLORIZE:
                            img_data = func_option(img_data, colors=self.colors, color_threshold=self.color_threshold)
                        else:
                            img_data = func_option(img_data)
                if self.file_format == ".zarr":
                    self._save_img_zarr(sample_path, img_name, img_data)
                elif self.file_format == ".png":
                    self._save_img_png(sample_path, img_name, img_data)
                else:
                    msg = f"Unknown file format ({self.file_format}) requested."
                    raise ValueError(msg)
        return (samples.shape[0] // self.sample_batch_size) * self.sample_batch_size
