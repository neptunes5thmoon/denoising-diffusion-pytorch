import os
from functools import partial
from pathlib import Path

import numpy as np
import zarr
from fibsem_tools import read, read_xarray
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from denoising_diffusion_pytorch.convenience import exists


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class SimpleDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        load_to_ram=False,
        convert_image_to=None,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        self.load_to_ram = load_to_ram
        if self.load_to_ram:
            self.imgs = [Image.open(path) for path in self.paths]
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.RandomCrop(image_size, padding=0),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.load_to_ram:
            img = self.imgs[index]
        else:
            path = self.paths[index]
            img = Image.open(path)
        return self.transform(img)


class ZarrDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        load_to_ram=True,
    ):
        super().__init__()
        self.folder = folder
        self.array_paths = [p for p in Path(folder).glob("**/*.zarr")]
        self.image_size = image_size
        self.load_to_ram = load_to_ram
        self.shapes = []
        self.crops = []
        for ap in self.array_paths:
            zg = zarr.load(str(ap))
            crop = zg["label"]
            self.shapes.append(crop.shape)
            if self.load_to_ram:
                self.crops.append(crop)

        self.transform = T.Compose(
            [
                T.ToTensor(),
                # RandomNonEmptyCrop(image_size, padding=0),
                T.RandomCrop(image_size, padding=0),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                # T.ToTensor(),
            ]
        )
        self.flat_index = []
        for crop, sh in enumerate(self.shapes):
            this_crop = [(crop, z) for z in range(sh[-1])]
            self.flat_index.extend(this_crop)

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, index):
        crop_idx, z = self.flat_index[index]
        if self.load_to_ram:
            arr = self.crops[crop_idx]
        else:
            arr = zarr.load(str(self.array_paths[crop_idx]))["label"]
        img = arr[..., z].transpose((1, 2, 0))
        return self.transform(img)


class CellMapDataset:
    def __init__(self, data_path, class_list, scale, annotation_path=None, crop_list=None):
        self.data_path = data_path
        self.scale = scale
        self.class_list = class_list
        if annotation_path is None:
            self.annotation_path = data_path
        else:
            self.annotation_path = annotation_path
        self.crops = self._get_crop_list(crop_list)

    def _get_crop_list(self, crop_list=None):
        if crop_list is None:
            sample = read(self.annotation_path)
            crops = ()
            for ds in sample:
                ann = read(os.path.join(self.annotation_path, ds))
                if "annotation" in ann.attrs["cellmap"]:
                    crop = AnnotationCrop(self.annotation_path, ds, self.scale)
                    if all(crop.is_fully_annotated(class_name) for class_name in self.class_list):
                        crops.append(crop)
            return crops
        else:
            return [AnnotationCrop(self.annotation_path, crop_name, self.scale) for crop_name in crop_list]
            # list(ds.keys())


class AnnotationCrop:
    def __init__(self, data_path, crop_name, scale):
        self.data_path = data_path
        self.crop_name = crop_name
        self.scale = scale
        self.crop = read(os.path.join(self.data_path, self.crop_name))

    def __repr__(self):
        return f"AnnotationCrop {self.crop_name} at {self.data_path} with scale {self.scale}"

    def _infer_scale_level(self, cls_name):
        if cls_name not in self.crop:
            msg = f"{cls_name} not found in {self}"
            raise ValueError(msg)
        arr_path = os.path.join(self.data_path, self.crop_name, cls_name)
        cls_msarr = read_xarray(arr_path)
        scales = dict()
        for name, dtarr in cls_msarr.children.items():
            arr = dtarr.data
            voxel_size = {ax: arr[ax].values[1] - arr[ax].values[0] for ax in arr.dims}
            if voxel_size == self.scale:
                return name
            scales[name] = voxel_size
        msg = f"{arr_path} does not contain array with voxel_size {self.scale}. Available scale levels are: {scales}"
        raise ValueError(msg)

    def get_classes(self):
        return self.crop.attrs["cellmap"]["annotation"]["class_names"]

    def get_class_array(self, cls_name):
        scale_level = self._infer_scale_level(cls_name)
        return self.crop[os.path.join(cls_name, scale_level)]

    def get_histogram(self, cls_name):
        return self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["histogram"]

    def get_possibilities(self, cls_name):
        return set(self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["annotation_type"]["encoding"].keys())

    def get_present_count(self, cls_name):
        try:
            class_array = self.get_class_array(cls_name)
        except ValueError:
            return 0
        histogram = self.get_histogram(cls_name)
        possiblities = self.get_possibilities(cls_name)
        possiblities.remove("present")
        not_present_sum = 0
        for possibility in possiblities:
            if possibility in histogram:
                not_present_sum += histogram[possibility]
        return np.prod(class_array.shape) - not_present_sum

    def has_present(self, cls_name):
        return self.get_present_count(cls_name) > 0

    def is_fully_annotated(self, cls_name):
        try:
            histogram = self.get_histogram(cls_name)
        except ValueError:
            return False
        if "unkown" not in histogram:
            return True
        else:
            return histogram["unkown"] == 0
