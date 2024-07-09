from __future__ import annotations

import logging
import os
from enum import Enum
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import dask
import datatree
import numpy as np
import torch
import xarray as xr
import zarr
from datatree import DataTree
from fibsem_tools import read, read_xarray
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, Dataset
from torchvision import utils
from torchvision.transforms import v2 as T

from denoising_diffusion_pytorch.convenience import exists

logger = logging.getLogger(__name__)


def get_nested_attr(attrs, key: str | Sequence[str | int]) -> Any:
    key_list: Sequence[str | int]
    if isinstance(key, str):
        key_list = key.split("/")
    else:
        key_list = key
    if len(key_list) == 1:
        return attrs[key_list[0]]
    else:
        return get_nested_attr(attrs[key_list[0]], key_list[1:])


def has_nested_attr(attrs, key: str | Sequence[str | int]) -> bool:
    key_list: Sequence[str | int]
    if isinstance(key, str):
        key_list = key.split("/")
    else:
        key_list = key
    if isinstance(key_list[0], str):
        if key_list[0] in attrs:
            return len(key_list) == 1 or has_nested_attr(attrs[key_list[0]], key_list[1:])
        else:
            return False
    elif isinstance(key_list[0], int):
        if len(attrs) > key_list[0]:
            return len(key_list) == 1 or has_nested_attr(attrs[key_list[0]], key_list[1:])
        else:
            return False
    else:
        msg = f"cannot handle key element {key_list[0]} of type {type(key_list[0])}"
        raise TypeError(msg)


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
                T.RandomCrop(image_size, padding=0) if image_size is not None else nn.Identity(),
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
        return self.transform(img) * 2. - 1.


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


class LabelRepresentation(str, Enum):
    BINARY = "binary"
    ONE_HOT = "one_hot"
    CLASS_IDS = "class_ids"


class RawChannelOptions(str, Enum):
    APPEND = "append"
    PREPEND = "prepend"
    FIRST = "first"
    SECOND = "second"
    EXCLUDE = "exclude"


class CellMapDatasets3Das2D(ConcatDataset):
    def __init__(
        self,
        data_paths: Sequence[str],
        class_list: Sequence[str],
        image_size: int,
        scale: dict[str, int],
        *,
        augment_horizontal_flip: bool = True,
        augment_vertical_flip: bool = True,
        annotation_paths: Sequence[str | None] | None = None,
        allow_single_class_crops: Sequence[str | None] | None = None,  # only has an effect if crop_lists is None
        crop_lists: Sequence[Sequence[str | None]] | None = None,
        raw_datasets: Sequence[str | None] | None = None,
        dask_workers: int = 0,
        pre_load: bool = False,
        contrast_adjust: bool = True,
        raw_channel: RawChannelOptions = RawChannelOptions.APPEND,
        label_representation: LabelRepresentation = LabelRepresentation.BINARY,
        random_crop: bool = True,
    ):
        cellmap_datasets = []
        if annotation_paths is None:
            annotation_paths = [
                None,
            ] * len(data_paths)
        if crop_lists is None:
            crop_lists = [
                None,
            ] * len(data_paths)
        if raw_datasets is None:
            raw_datasets = [
                "volumes/raw",
            ] * len(data_paths)
        assert len(data_paths) == len(annotation_paths) == len(crop_lists) == len(raw_datasets)
        for dp, ap, cl, rd in zip(data_paths, annotation_paths, crop_lists, raw_datasets):
            cellmap_datasets.append(
                CellMapDataset3Das2D(
                    dp,
                    class_list,
                    image_size,
                    scale,
                    augment_horizontal_flip=augment_horizontal_flip,
                    augment_vertical_flip=augment_vertical_flip,
                    allow_single_class_crops=allow_single_class_crops,
                    annotation_path=ap,
                    crop_list=cl,
                    raw_dataset=rd,
                    dask_workers=dask_workers,
                    pre_load=pre_load,
                    contrast_adjust=contrast_adjust,
                    raw_channel=raw_channel,
                    label_representation=label_representation,
                    random_crop=random_crop,
                )
            )
        super().__init__(cellmap_datasets)


class CellMapDataset3Das2D(ConcatDataset):
    def __init__(
        self,
        data_path: str,
        class_list: Sequence[str],
        image_size: int,
        scale: dict[str, int],
        *,
        augment_horizontal_flip: bool = True,
        augment_vertical_flip: bool = True,
        allow_single_class_crops: (
            Sequence[str | None] | None
        ) = None,  # only has an effect if crop_list is not specified
        annotation_path: str | None = None,
        crop_list: Sequence[str] | None = None,
        raw_dataset: str | None = "volumes/raw",
        dask_workers=0,
        pre_load=False,
        contrast_adjust=True,
        raw_channel: RawChannelOptions = RawChannelOptions.APPEND,
        label_representation: LabelRepresentation = LabelRepresentation.BINARY,
        random_crop: bool = True,
    ) -> None:
        self.pre_load = pre_load
        self.contrast_adjust = contrast_adjust
        self.data_path = data_path
        self.raw_dataset = raw_dataset
        self.scale = scale
        self.class_list = class_list
        self.image_size = image_size
        self._raw_scale: str | None = None
        self.augment_horizontal_flip = augment_horizontal_flip
        self.augment_vertical_flip = augment_vertical_flip
        self.dask_workers = dask_workers
        self.raw_channel = raw_channel
        self.label_representation = label_representation
        self.random_crop = random_crop
        self._raw_xarray: xr.DataArray | None = None
        if annotation_path is None:
            self.annotation_path = data_path
        else:
            self.annotation_path = annotation_path
        self.allow_single_class_crops: set[None | str] = set()
        if allow_single_class_crops is not None:
            self.allow_single_class_crops = set(self.allow_single_class_crops)
            if not self.allow_single_class_crops.issubset(set(self.class_list).union({None})):
                msg = f"`allow_single_class_crops` ({self.allow_single_class_crops}) should be subset of `class_list` ({self.class_list}) and {{None}}."
                raise ValueError(msg)
        self.crops = self._get_crop_list(crop_list)

        super().__init__(self.crops)
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
            ]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.data_path} at {self.scale} with crops {[c.crop_name for c in self.crops]}"

    def _get_crop_list(self, crop_list: Sequence[str] | None = None) -> list[AnnotationCrop3Das2D]:
        filtering = self.allow_single_class_crops != set(self.class_list).union({None})
        if crop_list is None:
            sample = read(self.annotation_path)
            crops = []
            for ds in sample:
                ann = read(os.path.join(self.annotation_path, ds))
                if has_nested_attr(ann.attrs, ["cellmap", "annotation"]):
                    crop = AnnotationCrop3Das2D(
                        self,
                        self.annotation_path,
                        ds,
                        dask_workers=self.dask_workers,
                        pre_load=self.pre_load,
                        contrast_adjust=self.contrast_adjust,
                        raw_channel=self.raw_channel,
                        label_representation=self.label_representation,
                    )
                    if all(crop.sizes[dim] >= self.image_size for dim in ["x", "y"]) and all(
                        crop.is_fully_annotated(class_name) for class_name in self.class_list
                    ):
                        if filtering:
                            present_classes = {
                                class_name for class_name in self.class_list if crop.has_present(class_name)
                            }
                            if (
                                len(present_classes) > 1
                                or (
                                    len(present_classes) == 1
                                    and present_classes.issubset(self.allow_single_class_crops)
                                )
                                or (len(present_classes) == 0 and None in self.allow_single_class_crops)
                            ):
                                crops.append(crop)
                        else:
                            crops.append(crop)
                    else:
                        if all(crop.sizes[dim] >= self.image_size for dim in ["x", "y"]):
                            msg = f"{crop} has sizes {crop.sizes}, which is too small for patch size {self.image_size}"
                        else:
                            not_incl = []
                            for class_name in self.class_list:
                                if not crop.is_fully_annotated(class_name):
                                    not_incl.append(class_name)
                            msg = f"{not_incl} not annotated in {crop}"
                        logger.debug(msg)
        else:
            crops = [
                AnnotationCrop3Das2D(
                    self,
                    self.annotation_path,
                    crop_name,
                    dask_workers=self.dask_workers,
                    pre_load=self.pre_load,
                    contrast_adjust=self.contrast_adjust,
                    raw_channel=self.raw_channel,
                    label_representation=self.label_representation,
                    random_crop=self.random_crop,
                )
                for crop_name in crop_list
            ]
        if len(crops) == 0:
            msg = f"List of crops for {self.data_path} with annotations for {self.class_list} at {self.annotation_path} is empty."
            raise ValueError(msg)
        return crops

    @property
    def raw_xarray(self) -> None | xr.DataArray | DataTree:
        if self.raw_dataset is None:
            return None
        if self._raw_xarray is None:
            self._raw_xarray = read_xarray(os.path.join(self.data_path, self.raw_dataset, self.raw_scale))
        return self._raw_xarray

    @property
    def raw_scale(self) -> str | None:
        if self.raw_dataset is None:
            return None
        if self._raw_scale is None:
            arr_path = os.path.join(self.data_path, self.raw_dataset)
            msarr = read_xarray(arr_path)
            if isinstance(msarr, datatree.DataTree):
                scale_to_voxelsize = {}
                for name, dtarr in msarr.children.items():
                    arr = dtarr.data
                    voxel_size = {ax: arr[ax].values[1] - arr[ax].values[0] for ax in arr.dims}
                    if voxel_size == self.scale:
                        self._raw_scale = name
                        break
                    scale_to_voxelsize[name] = voxel_size
                if self._raw_scale is None:
                    msg = f"{arr_path} does not contain array with voxel_size {self.scale}. Available scale levels are: {scale_to_voxelsize}"
                    raise ValueError(msg)
            else:
                voxel_size = {ax: msarr[ax].values[1] - msarr[ax].values[0] for ax in msarr.dims}
                if voxel_size != self.scale:
                    msg = f"{arr_path} has scale {voxel_size}, asked for scale {self.scale}"
                    raise ValueError(msg)
                self._raw_scale = ''
        return self._raw_scale

    def __getitem__(self, idx: int) -> Tensor:
        # img = super().__getitem__(idx)
        # if isinstance(img, tuple):
        #     img_tensor = []
        #     for im in img:
        #         img_tensor.append(totensor(im))
        #     img_tensor = tuple(img_tensor)
        # else:
        #     img_tensor = totensor(img)
        # return self.transform(img_tensor)
        return self.transform(super().__getitem__(idx))

    #     for name, dtarr in cls_msarr.children.items():
    #         arr = dtarr.data
    #         voxel_size = {ax: arr[ax].values[1] - arr[ax].values[0] for ax in arr.dims}
    #         if voxel_size == self.parent_data.scale:
    #             return name
    #         scale_to_voxelsize[cls_name] = voxel_size
    #     msg = f"{arr_path} does not contain array with voxel_size {self.parent_data.scale}. Available scale levels are: {scale_to_voxelsize}"
    #     raise ValueError(msg)


class AnnotationCrop3Das2D(Dataset):
    def __init__(
        self,
        parent_data: CellMapDataset3Das2D,
        annotation_path: str,
        crop_name: str,
        dask_workers: int = 0,
        pre_load=False,
        contrast_adjust=True,
        raw_channel: RawChannelOptions = "APPEND",
        label_representation: LabelRepresentation = "BINARY",
        random_crop: bool = True,
    ):
        self.parent_data = parent_data
        self.annotation_path = annotation_path
        self.crop_name = crop_name
        self.crop: = read(os.path.join(self.annotation_path, self.crop_name))
        if not has_nested_attr(self.crop["labels"].attrs, ["cellmap", "annotation"]):
            msg = f"Crop {crop_name} at {annotation_path} is not a cellmap annotation crop."
            raise ValueError(msg)
        self._scales: dict[str, str] | None = None
        self._sizes: None | Mapping[str, int] = None
        self._size: None | int = None
        self._coords: None | xr.Coordinates = None
        self.annotated_classes = get_nested_attr(self.crop["labels"].attrs, ["cellmap", "annotation", "class_names"])
        self.class_list = list(set(self.annotated_classes).intersection(set(self.parent_data.class_list)))
        self._class_xarray: dict[str, xr.DataArray] = dict()
        self._raw_xarray = None
        self._class_ids_xarray = None
        self.dask_workers = dask_workers
        self.pre_load = pre_load
        self.contrast_adjust = contrast_adjust
        self._contrast_min = None
        self._contrast_max = None
        self.raw_channel = raw_channel
        self.label_representation = label_representation
        self.random_crop = random_crop

    @property
    def raw_xarray(self):
        if self._raw_xarray is None:
            if "raw" in self.crop:
                try:
                    mslvl = self._infer_scale_level("raw")

                    self._raw_xarray = read_xarray(
                        os.path.join(self.annotation_path, self.crop_name, "raw", mslvl), use_dask=not self.pre_load
                    )
                    # if self.pre_load:
                    #    self._raw_xarray = self._raw_xarray.compute(workers = self.dask_workers)
                except ValueError as e:
                    if self.parent_data.raw_dataset is None:
                        msg = "Parent raw dataset is not set and no raw data found in crop"
                        raise ValueError(msg) from e
                    self._raw_xarray = self.parent_data.raw_xarray.copy()
            else:
                if self.parent_data.raw_dataset is None:
                    msg = "Parent raw dataset is not set and no raw data found in crop"
                    raise ValueError(msg)
                self._raw_xarray = self.parent_data.raw_xarray.copy()
        return self._raw_xarray

    @property
    def contrast_min(self):
        if self._contrast_min is None:
            if "raw" in self.crop:
                raw = read_xarray(os.path.join(self.annotation_path, self.crop_name, "raw"))
            else:
                if self.parent_data.raw_dataset is None:
                    msg = "Parent raw dataset is not set and no raw data found in crop"
                    raise ValueError(msg)
                raw = read_xarray(os.path.join(self.parent_data.data_path, self.parent_data.raw_dataset))
            if has_nested_attr(raw.attrs, ["cellmap", "contrast", "min"]):
                self._contrast_min = get_nested_attr(raw.attrs, ["cellmap", "contrast", "min"])
            elif has_nested_attr(raw.attrs, ["contrastAdjustment", "min"]):
                self._contrast_min = get_nested_attr(raw.attrs, ["contrastAdjustment", "min"])
            else:
                logger.debug("Defaulting min of contrast adjustment to 0.")
                self._contrast_min = 0
        return self._contrast_min

    @property
    def contrast_max(self):
        if self._contrast_max is None:
            if "raw" in self.crop:
                raw = read_xarray(os.path.join(self.annotation_path, self.crop_name, "raw"))
            else:
                if self.parent_data.raw_dataset is None:
                    msg = "Parent raw dataset is not set and no raw data found in crop"
                    raise ValueError(msg)
                raw = read_xarray(os.path.join(self.parent_data.data_path, self.parent_data.raw_dataset))
            if has_nested_attr(raw.attrs, ["cellmap", "contrast", "max"]):
                self._contrast_max = get_nested_attr(raw.attrs, ["cellmap", "contrast", "max"])
            elif has_nested_attr(raw.attrs, ["contrastAdjustment", "max"]):
                self._contrast_max = get_nested_attr(raw.attrs, ["contrastAdjustment", "max"])
            else:
                logger.debug("Defaulting max of contrast adjustment to 255.")
                self._contrast_max = 255
        return self._contrast_max

    @property
    def scales(self) -> dict[str, str]:
        if self._scales is None:
            self._scales = {}
            for class_name in self.annotated_classes:
                self._scales[class_name] = self._infer_scale_level(f"labels/{class_name}")
        return self._scales

    @property
    def sizes(self) -> Mapping[str, int]:
        if self._sizes is None:
            self._sizes = self.get_xarray_attr("sizes")
        return self._sizes

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self.get_xarray_attr("size")
        return self._size

    @property
    def coords(self) -> xr.Coordinates:
        if self._coords is None:
            self._coords = self.get_xarray_attr("coords")
        return self._coords

    def get_xarray_attr(self, attr: str) -> Any:
        ref_attr = None
        for class_name in self.class_list:
            curr_attr = getattr(self.class_xarray(class_name), attr)
            if ref_attr is None:
                ref_attr = curr_attr
            elif curr_attr != ref_attr:
                msg = f"Crop {self} has arrays with different values for {attr} for requested scale {self.parent_data.scale}. Found (at least) {curr_attr}, {ref_attr}."
                raise ValueError(msg)
        return ref_attr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.crop_name} at {self.annotation_path} from {self.parent_data.__class__.__name__} at {self.parent_data.data_path}"

    def _infer_scale_level(self, ds_name: str) -> str:
        if ds_name not in self.crop:
            msg = f"{ds_name} not found in {self}"
            raise ValueError(msg)
        arr_path = os.path.join(self.annotation_path, self.crop_name, ds_name)
        msarr = read_xarray(arr_path)
        scale_to_voxelsize = dict()
        for name, dtarr in msarr.children.items():
            arr = dtarr.data
            voxel_size = {ax: arr[ax].values[1] - arr[ax].values[0] for ax in arr.dims}
            if voxel_size == self.parent_data.scale:
                return name
            scale_to_voxelsize[ds_name] = voxel_size
        msg = f"{arr_path} does not contain array with voxel_size {self.parent_data.scale}. Available scale levels are: {scale_to_voxelsize}"
        raise ValueError(msg)

    def get_classes(self) -> list[str]:
        return get_nested_attr(self.crop["labels"].attrs, ["cellmap", "annotation", "class_names"])

    def get_class_array(self, cls_name: str):
        if cls_name not in self.annotated_classes:
            msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
            raise ValueError(msg)
        return self.crop[os.path.join("labels", cls_name, self.scales[cls_name])]

    def class_xarray(self, cls_name: str) -> xr.DataArray:
        if cls_name not in self._class_xarray:
            if cls_name == "background" and "background" not in self.annotated_classes:
                arrs = []
                for cls_iter in self.parent_data.class_list:
                    cls_arr = self.class_xarray(cls_iter)
                    arrs.append(cls_arr)
                self._class_xarray[cls_name] = xr.ones_like(arrs[-1]) - np.sum(arrs, axis=0)
            elif cls_name not in self.annotated_classes:
                msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
                raise ValueError(msg)
            else:
                full_path = os.path.join(
                    self.annotation_path, self.crop_name, "labels", cls_name, self.scales[cls_name]
                )
                self._class_xarray[cls_name] = read_xarray(full_path, name=full_path, use_dask=not self.pre_load)  # type: ignore
            # if self.pre_load:
            #    self._class_xarray[cls_name] = self._class_xarray[cls_name].compute(workers=self.dask_workers)
        return self._class_xarray[cls_name]

    @property
    def class_ids_xarray(self):
        if self._class_ids_xarray is None:
            arrs = []
            for cls_name in self.parent_data.class_list:
                cls_arr = self.class_xarray(cls_name)
                arrs.append(cls_arr)
            arrs = [xr.zeros_like(arrs[0]), *arrs]
            class_ids_arr = xr.concat(arrs, dim="class").argmax(axis=0)
            if self.pre_load:
                self._class_ids_xarray = class_ids_arr.compute(workers=self.dask_workers)
            else:
                self._class_ids_xarray = class_ids_arr
        return self._class_ids_xarray

    def get_counts(self, cls_name: str) -> Mapping[str, int]:
        return get_nested_attr(self.get_class_array(cls_name).attrs, ["cellmap", "annotation", "complement_counts"])

    def get_possibilities(self, cls_name: str) -> set[str]:
        return set(
            get_nested_attr(
                self.get_class_array(cls_name).attrs, ["cellmap", "annotation", "annotation_type", "encoding"]
            ).keys()
        )

    def get_present_count(self, cls_name: str) -> int:
        counts = self.get_counts(cls_name)
        possiblities = self.get_possibilities(cls_name)
        possiblities.remove("present")
        not_present_sum = 0
        for possibility in possiblities:
            if possibility in counts:
                not_present_sum += counts[possibility]
        return self.size - not_present_sum

    def has_present(self, cls_name: str) -> bool:
        return self.get_present_count(cls_name) > 0

    def is_fully_annotated(self, cls_name: str) -> bool:
        if cls_name not in self.annotated_classes:
            return False
        counts = self.get_counts(cls_name)
        if "unknown" not in counts:
            return True
        else:
            return counts["unknown"] == 0

    def __len__(self):
        if self.random_crop:
            return self.sizes["z"]
        else:
            return np.prod(
                (
                    self.sizes["z"],
                    int(np.ceil(self.sizes["x"] / self.parent_data.image_size)),
                    int(np.ceil(self.sizes["y"] / self.parent_data.image_size)),
                )
            )

    def __getitem__(self, idx: int) -> np.ndarray:
        if self.random_crop:
            x_start = np.random.randint(0, self.sizes["x"] - self.parent_data.image_size + 1)
            y_start = np.random.randint(0, self.sizes["y"] - self.parent_data.image_size + 1)
            vox_slice = {
                "z": idx,
                "x": slice(x_start, x_start + self.parent_data.image_size),
                "y": slice(y_start, y_start + self.parent_data.image_size),
            }
        else:
            idx_tuple = np.unravel_index(
                idx,
                (
                    self.sizes["z"],
                    int(np.ceil(self.sizes["y"] / self.parent_data.image_size)),
                    int(np.ceil(self.sizes["x"] / self.parent_data.image_size)),
                ),
            )
            vox_slice = {
                "z": idx_tuple[0],
                "y": slice(
                    idx_tuple[1] * self.parent_data.image_size, (idx_tuple[1] + 1) * self.parent_data.image_size
                ),
                "x": slice(
                    idx_tuple[2] * self.parent_data.image_size, (idx_tuple[2] + 1) * self.parent_data.image_size
                ),
            }

        arrs: list[xr.DataArray]
        if self.label_representation == LabelRepresentation.CLASS_IDS:
            arrs = [self.class_ids_xarray.isel(vox_slice)]
        else:
            arrs = []
            if self.label_representation == LabelRepresentation.ONE_HOT:
                bg_arr = self.class_xarray("background").isel(vox_slice)
                arrs.append(bg_arr.astype("float32"))
            for cls_name in self.parent_data.class_list:
                cls_arr = self.class_xarray(cls_name).isel(vox_slice)
                arrs.append(cls_arr.astype("float32"))

        if self.raw_channel == RawChannelOptions.EXCLUDE:
            for k, arr in enumerate(arrs):
                if arr.sizes["x"] < self.parent_data.image_size or arr.sizes["y"] < self.parent_data.image_size:
                    arrs[k] = arr.pad(
                        pad_with={
                            "x": (0, self.parent_data.image_size - arr.sizes["x"]),
                            "y": (0, self.parent_data.image_size - arr.sizes["y"]),
                        }
                    )

            res = dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers)
        else:
            spatial_slice = {
                dim: slice(
                    int(arrs[-1].coords[dim][0])
                    - self.parent_data.scale[dim] / 2,  # hack to deal with misalignment/wrong offsets
                    int(arrs[-1].coords[dim][-1]) + self.parent_data.scale[dim] / 2,
                )
                for dim in "xy"
            }
            spatial_slice["z"] = slice(
                int(arrs[-1].coords["z"]) - self.parent_data.scale["z"] / 2,
                int(arrs[-1].coords["z"]) + self.parent_data.scale["z"] / 2,
            )
            raw_arr = self.raw_xarray.sel(spatial_slice).squeeze().astype('float32')
            if self.contrast_adjust:
                raw_arr = (raw_arr - self.contrast_min) / (self.contrast_max - self.contrast_min)
            else:
                raw_arr = raw_arr / 255.0
            raw_arr = (raw_arr * 2.0) - 1.0
            for k, arr in enumerate(arrs):
                if arr.sizes["x"] < self.parent_data.image_size or arr.sizes["y"] < self.parent_data.image_size:
                    arrs[k] = np.pad(
                        arr.data,
                        pad_width=[
                            (0, self.parent_data.image_size - arr.data.shape[0]),
                            (0, self.parent_data.image_size - arr.data.shape[1]),
                        ],
                    )
                else:
                    arrs[k] = arr.data

            if raw_arr.shape[0] < self.parent_data.image_size or raw_arr.shape[1] < self.parent_data.image_size:
                raw_arr = np.pad(
                    raw_arr,
                    [
                        (0, self.parent_data.image_size - raw_arr.shape[0]),
                        (0, self.parent_data.image_size - raw_arr.shape[1]),
                    ],
                )
            if self.raw_channel == RawChannelOptions.APPEND:
                arrs.append(raw_arr)
                res = dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers)
            elif self.raw_channel == RawChannelOptions.PREPEND:
                arrs = [raw_arr, *arrs]
                res = dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers)
            elif self.raw_channel == RawChannelOptions.FIRST:
                res = (
                    np.expand_dims(raw_arr.data, -1),
                    dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers),
                )
            elif self.raw_channel == RawChannelOptions.SECOND:
                res = (
                    dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers),
                    np.expand_dims(raw_arr.data, -1),
                )
            else:
                msg = f"Unknown option for handling raw channel: {self.raw_channel}"
                raise ValueError(msg)
        return (
            res  # shape (self.parent_data.image_size, self.parent_data.image_size, len(self.parent_data.class_list)+1)
        )


class BatchedZarrSamples(Dataset):
    def __init__(
        self,
        zarr_path: str,
        labels: str = "labels",
        raw: str = "raw",
        raw_channel: RawChannelOptions = RawChannelOptions.APPEND,
        **kwargs,
        # label_representation_in: LabelRepresentation = LabelRepresentation.BINARY,
        # label_representation_out: LabelRepresentation = LabelRepresentation.BINARY,
    ):
        self.zarr_path = zarr_path
        self.zarr_file = zarr.open(self.zarr_path, "r")
        self.digits = len(list(self.zarr_file.keys())[0])

        for k in self.zarr_file.keys():
            if self.digits != len(k):
                msg = f"Keys of {self.zarr_path} do not have same length ({self.digits})"
                raise ValueError(msg)
        self.raw_channel = raw_channel
        # self.label_representation_in = label_representation_in
        # self.label_representation_out = label_representation_out
        # if labels is None:
        # self.labels = [
        #         "ecs",
        #         "pm",
        #         "mito_mem",
        #         "mito_lum",
        #         "mito_ribo",
        #         "golgi_mem",
        #         "golgi_lum",
        #         "ves_mem",
        #         "ves_lum",
        #         "endo_mem",
        #         "endo_lum",
        #         "lyso_mem",
        #         "lyso_lum",
        #         "ld_mem",
        #         "ld_lum",
        #         "er_mem",
        #         "er_lum",
        #         "eres_mem",
        #         "eres_lum",
        #         "ne_mem",
        #         "ne_lum",
        #         "np_out",
        #         "np_in",
        #         "hchrom",
        #         "nhchrom",
        #         "echrom",
        #         "nechrom",
        #         "nucpl",
        #         "nucleo",
        #         "mt_out",
        #         "mt_in",
        #     ]
        # else:
        self.labels = labels
        self.raw = raw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.zarr_path}"

    def __len__(self):
        return len(self.zarr_file.keys()) - 1

    def __getitem__(self, idx: int) -> np.ndarray:
        sample_arr = self.zarr_file[f"{idx:0{self.digits}d}"]

        arrs: list[np.ndarray] = []
        arrs.append(np.array(sample_arr[self.labels]))
        # if self.label_representation_out == LabelRepresentation.CLASS_IDS:
        #     for cls_name in self.labels:
        #         arrs.append(np.array(sample_arr[cls_name]))
        #     arrs = [np.zeros_like(arrs[0]), *arrs]
        #     arrs = np.concatenate(arrs, axis=0).argmax(axis=0)
        # else:
        #     if self.label_representation == LabelRepresentation.ONE_HOT:
        #         if "background" not in self.labels:
        #             bg_arrs = []
        #             for cls_iter in self.labels:
        #                 bg_arrs.append(np.array(sample_arr[cls_iter]))
        #             arrs.append(np.ones_like(arrs[-1]) - np.sum(arrs, axis=0))
        #     for cls_name in self.labels:
        #         arrs.append(np.array(sample_arr[cls_name]))

        if self.raw_channel == RawChannelOptions.EXCLUDE:
            res = np.concatenate(arrs, axis=1)
        else:
            raw_arr = (np.array(sample_arr["raw"]) / 255.0).astype(np.float32)
            raw_arr = (raw_arr * 2.0) - 1.0
            if self.raw_channel == RawChannelOptions.APPEND:
                arrs.append(raw_arr)
                res = np.concatenate(arrs, axis=1)
            elif self.raw_channel == RawChannelOptions.PREPEND:
                arrs = [raw_arr, *arrs]
                res = np.concatenate(arrs, axis=1)
            elif self.raw_channel == RawChannelOptions.FIRST:
                res = (raw_arr, np.concatenate(arrs, axis=1))
            elif self.raw_channel == RawChannelOptions.SECOND:
                res = (
                    np.concatenate(arrs, axis=1),
                    raw_arr,
                )
            else:
                msg = f"Unknown option for handling raw channel: {self.raw_channel}"
                raise ValueError(msg)
        return res


if __name__ == "__main__":
    data_path = "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.n5"
    annotation_path = "/nrs/cellmap/data/jrc_hela-2/staging/groundtruth.zarr"
    ds = CellMapDataset3Das2D(
        data_path,
        [
            "ecs",
            "pm",
            "mito_mem",
            "mito_lum",
            "mito_ribo",
            "golgi_mem",
            "golgi_lum",
            "ves_mem",
            "ves_lum",
            "endo_mem",
            "endo_lum",
            "lyso_mem",
            "lyso_lum",
            "ld_mem",
            "ld_lum",
            "er_mem",
            "er_lum",
            "eres_mem",
            "eres_lum",
            "ne_mem",
            "ne_lum",
            "np_out",
            "np_in",
            "hchrom",
            "nhchrom",
            "echrom",
            "nechrom",
            "nucpl",
            "nucleo",
            "mt_out",
            "mt_in",
        ],
        64,
        {"x": 4, "y": 4, "z": 4},
        annotation_path=annotation_path,
        crop_list=['crop1', 'crop113'],
    )
    print(ds)
    print(len(ds))
    print(type(ds[10]))
    print(ds[10].shape)
    print(ds[10])
