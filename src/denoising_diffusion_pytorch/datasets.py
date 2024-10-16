from __future__ import annotations

import logging
import os
import yaml
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import dask
import datatree
import numpy as np
import xarray as xr
import zarr
from datatree import DataTree
from fibsem_tools import read

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2 as transforms
import cellmap_utils_kit
from cellmap_utils_kit.h5_xarray_reader import read_any_xarray
from denoising_diffusion_pytorch.convenience import exists
import json

logger = logging.getLogger(__name__)


def get_nested_attr(attrs, key: str | Sequence[str | int]) -> Any:
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
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
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
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
        exts=("jpg", "jpeg", "png", "tiff"),
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

        self.transform = transforms.Compose(
            [
                transforms.Lambda(maybe_convert_fn),
                transforms.RandomCrop(image_size, padding=0) if image_size is not None else nn.Identity(),
                transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                transforms.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                transforms.ToTensor(),
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
        return self.transform(img) * 2.0 - 1.0


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
        self.array_paths = list(Path(folder).glob("**/*.zarr"))
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

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # RandomNonEmptyCrop(image_size, padding=0),
                transforms.RandomCrop(image_size, padding=0),
                transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                transforms.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                # transforms.ToTensor(),
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
    CLASS_IDS = "class_ids" # labels are combined into a single array, each class has a unique id


class RawChannelOptions(str, Enum):
    APPEND = "append" # append in the channel dimension as last channel
    PREPEND = "prepend" # prepend in the channel dimension as first channel
    FIRST = "first" # return tensor with raw data as first argument
    SECOND = "second" # return tensor with raw data as second argument
    EXCLUDE = "exclude" # don't return raw data

class ClassOptions(str, Enum):
    DATASET = "dataset"

class CellMapDatasets3Das2D(ConcatDataset):
    def __init__(
        self,
        data_config: str,
        class_list: Sequence[str],
        image_size: int,
        scale: dict[str, int],
        *,
        augment_horizontal_flip: bool = True,
        augment_vertical_flip: bool = True,
        dask_workers: int = 0,
        pre_load: bool = False,
        contrast_adjust: bool = True,
        raw_channel: RawChannelOptions = RawChannelOptions.APPEND,
        label_representation: LabelRepresentation = LabelRepresentation.BINARY,
        random_crop: bool = True,
        classes: ClassOptions | None = None
    ):
        cellmap_datasets = []
        with open(data_config) as f:
            datasets = yaml.safe_load(f)["datasets"]
        self.classes = classes
        self.class_idx_to_name = {}
        for dataset_idx, (dataname, datainfo) in enumerate(datasets.items()):
            cellmap_datasets.append(
                CellMapDataset3Das2D(
                    dataname,
                    datainfo,
                    dataset_idx,
                    class_list,
                    image_size,
                    scale,
                    augment_horizontal_flip=augment_horizontal_flip,
                    augment_vertical_flip=augment_vertical_flip,
                    dask_workers=dask_workers,
                    pre_load=pre_load,
                    contrast_adjust=contrast_adjust,
                    raw_channel=raw_channel,
                    label_representation=label_representation,
                    random_crop=random_crop,
                    classes=classes
                )
            )
            if classes == ClassOptions.DATASET:
                self.class_idx_to_name[dataset_idx] = dataname
        super().__init__(cellmap_datasets)


class CellMapDataset3Das2D(ConcatDataset):
    def __init__(
        self,
        dataname: str,
        datainfo: dict,
        dataset_idx: int,
        class_list: Sequence[str],
        image_size: int,
        scale: dict[str, int],
        *,
        augment_horizontal_flip: bool = True,
        augment_vertical_flip: bool = True,
        dask_workers=0,
        pre_load=False,
        contrast_adjust=True,
        raw_channel: RawChannelOptions = RawChannelOptions.APPEND,
        label_representation: LabelRepresentation = LabelRepresentation.BINARY,
        random_crop: bool = True,
        label_subgroup: str = "labels",
        classes: ClassOptions | None = None,
    ) -> None:
        self.pre_load = pre_load
        self.contrast_adjust = contrast_adjust
        self.dataname = dataname
        self.datainfo = datainfo
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
        self.allow_single_class_crops: set[None | str] = set()
        self.label_subgroup = label_subgroup
        self.classes = classes
        self.dataset_idx = dataset_idx

        # construct list of crops
        self.crops = self._get_crop_list(datainfo)
        # make the concatenated torch.Dataset
        super().__init__(self.crops)

        self.transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                transforms.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
            ]
        )

    def __repr__(self) -> str:
        descr = (
            f"{self.__class__.__name__}: {self.dataname} at {self.scale} "
            f"with crops {[c.crop_name for c in self.crops]}"
        )
        return descr

    def _get_crop_list(self, datainfo) -> list[AnnotationCrop3Das2D]:
        crops = [
                AnnotationCrop3Das2D(
                    self,
                    datainfo["crop_group"],
                    crop_name,
                    dask_workers=self.dask_workers,
                    pre_load=self.pre_load,
                    contrast_adjust=self.contrast_adjust,
                    raw_channel=self.raw_channel,
                    label_representation=self.label_representation,
                    random_crop=self.random_crop,
                    label_subgroup=self.label_subgroup,
                    classes = self.classes
                )
                for crop_name in datainfo["crops"]
            ]
        if len(crops) == 0:
            msg = (
                f"List of crops for {self.data_path} with annotations for {self.class_list}"
                f"at {self.annotation_path} is empty."
            )
            raise ValueError(msg)
        return crops

    @property
    def raw_xarray(self) -> None | xr.DataArray | DataTree:
        if "raw" not in self.datainfo or self.datainfo["raw"] is None:
            return None
        if self._raw_xarray is None:
            self._raw_xarray = read_any_xarray(Path(self.datainfo["raw"]) / self.raw_scale)
        return self._raw_xarray

    @property
    def raw_scale(self) -> str | None:
        if "raw" not in self.datainfo or self.datainfo["raw"] is None:
            return None
        if self._raw_scale is None:
            arr_path = Path(self.datainfo["raw"])
            self._raw_scale = cellmap_utils_kit.attribute_handler.get_scalelevel(read(arr_path))
        return self._raw_scale

    def __getitem__(self, idx: int) -> Tensor:
        return self.transform(super().__getitem__(idx))

class AnnotationCrop3Das2D(Dataset):
    def __init__(
        self,
        parent_data: CellMapDataset3Das2D,
        crop_dir: str,
        crop_name: str,
        dask_workers: int = 0,
        pre_load=False,
        contrast_adjust=True,
        raw_channel: RawChannelOptions = "APPEND",
        label_representation: LabelRepresentation = "BINARY",
        random_crop: bool = True,
        label_subgroup: str = "labels",
        classes: ClassOptions | None = None
    ):
        """_summary_

        Args:
            parent_data (CellMapDataset3Das2D): _description_
            annotation_path (str): path to zarr that contains crop group
            crop_name (str): name of the crop (and accordingly of the crop group)
            dask_workers (int, optional): Number of workers to use for dask operations. Defaults to 0.
            pre_load (bool, optional): Whether to load data at initialization. Defaults to False.
            contrast_adjust (bool, optional): Whether to adjust contrast of raw data. Defaults to True.
            raw_channel (RawChannelOptions, optional): Defines what to do with the raw channel, options are "APPEND", "PREPEND", "FIRST", "SECOND" and "EXCLUDE". Defaults to "APPEND".
            label_representation (LabelRepresentation, optional): Defines how labels should be represented, options are "BINARY", "ONE_HOT" or "CLASS_IDS". Defaults to "BINARY".
            random_crop (bool, optional): _description_. Defaults to True.
            classes (ClassOptions, optional): If not None, defines what classes should be used for classifier free guidance. Options are "DATASET"

        Raises:
            ValueError: _description_
        """
        self.parent_data = parent_data
        self.crop_dir = Path(crop_dir)
        self.crop_name = crop_name
        self.crop = read(os.path.join(self.crop_dir, self.crop_name))
        if label_subgroup is None or label_subgroup == "":
            self.label_attrs = self.crop.attrs
        else:
            self.label_attrs = self.crop[label_subgroup].attrs
        if not has_nested_attr(self.label_attrs, ["cellmap", "annotation"]):
            msg = f"Crop {crop_name} at {self.crop_dir} is not a cellmap annotation crop."
            raise ValueError(msg)
        self._scales: dict[str, str] | None = None
        self._sizes: None | Mapping[str, int] = None
        self._size: None | int = None
        self._coords: None | xr.Coordinates = None
        self.annotated_classes = get_nested_attr(self.label_attrs, ["cellmap", "annotation", "class_names"])
        self.class_list = list(set(self.annotated_classes).intersection(set(self.parent_data.class_list)))
        self._class_xarray: dict[str, xr.DataArray] = {}
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
        self.label_subgroup = label_subgroup
        self.classes = classes

    @property
    def raw_xarray(self):
        if self._raw_xarray is None:
            if "raw" in self.crop:
                try:
                    mslvl = self._infer_scale_level("raw")
                    self._raw_xarray = read_any_xarray(
                        self.crop_dir / self.crop_name / "raw" / mslvl, use_dask=not self.pre_load
                    )
                except ValueError as e:
                    if self.parent_data.raw_xarray is None:
                        msg = "Raw data is not configured and no raw data found in crop"
                        raise ValueError(msg) from e
                    self._raw_xarray = self.parent_data.raw_xarray.copy()
            else:
                if self.parent_data.raw_xarray is None:
                    msg = "Raw data is not configured and no raw data found in crop"
                    raise ValueError(msg)
                self._raw_xarray = self.parent_data.raw_xarray.copy()
        return self._raw_xarray

    @property
    def contrast_min(self):
        if self._contrast_min is None:
            if "raw" in self.crop:
                raw = read_any_xarray(self.crop_dir / self.crop_name / "raw")
            else:
                if self.parent_data.raw_xarray is None:
                    msg = "Raw data is not configured and no raw data found in crop"
                    raise ValueError(msg)
                raw = read_any_xarray(self.parent_data.datainfo["raw"])
            if has_nested_attr(raw.attrs, ["cellmap", "contrast", "min"]):
                self._contrast_min = get_nested_attr(raw.attrs, ["cellmap", "contrast", "min"])
            elif has_nested_attr(raw.attrs, ["contrastAdjustment", "min"]):
                self._contrast_min = get_nested_attr(raw.attrs, ["contrastAdjustment", "min"])
            elif "raw" in self.crop:
                logger.debug("Defaulting min of contrast adjustmnet to min value in cropped raw data.")
                self._contrast_min = self.raw_xarray.min().compute()
            else:
                logger.debug("Defaulting min of contrast adjustment to 0.")
                self._contrast_min = 0
        return self._contrast_min

    @property
    def contrast_max(self):
        if self._contrast_max is None:
            if "raw" in self.crop:
                raw = read_any_xarray(self.crop_dir / self.crop_name / "raw")
            else:
                if self.parent_data.raw_xarray is None:
                    msg = "Raw data is not configured and no raw data found in crop"
                    raise ValueError(msg)
                raw = read_any_xarray(self.parent_data.datainfo["raw"])
            if has_nested_attr(raw.attrs, ["cellmap", "contrast", "max"]):
                self._contrast_max = get_nested_attr(raw.attrs, ["cellmap", "contrast", "max"])
            elif has_nested_attr(raw.attrs, ["contrastAdjustment", "max"]):
                self._contrast_max = get_nested_attr(raw.attrs, ["contrastAdjustment", "max"])
            elif "raw" in self.crop:
                logger.debug("Defaulting max of contrast adjustment to max value in cropped raw data.")
                self._contrast_max = self.raw_xarray.max().compute()
            else:
                logger.debug("Defaulting max of contrast adjustment to 255.")
                self._contrast_max = 0
        return self._contrast_max

    @property
    def scales(self) -> dict[str, str]:
        if self._scales is None:
            self._scales = {}
            for class_name in self.annotated_classes:
                self._scales[class_name] = self._infer_scale_level(os.path.join(self.label_subgroup, class_name))
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
                msg = (
                    f"Crop {self} has arrays with different values for {attr} "
                    f"for requested scale {self.parent_data.scale}. "
                    f"Found (at least) {curr_attr}, {ref_attr}."
                )
                raise ValueError(msg)
        return ref_attr

    def __repr__(self) -> str:
        descr = (
            f"{self.__class__.__name__} {self.crop_name} at {self.annotation_path} from "
            f"{self.parent_data.__class__.__name__} at {self.parent_data.data_path}"
        )
        return descr

    def _infer_scale_level(self, ds_name: str) -> str:
        return cellmap_utils_kit.attribute_handler.get_scalelevel(self.crop[ds_name],self.parent_data.scale)

    def get_classes(self) -> list[str]:
        return get_nested_attr(self.label_attrs, ["cellmap", "annotation", "class_names"])

    def get_class_array(self, cls_name: str):
        if cls_name not in self.annotated_classes:
            msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
            raise ValueError(msg)
        return self.crop[os.path.join(self.label_subgroup, cls_name, self.scales[cls_name])]

    def class_xarray(self, cls_name: str) -> xr.DataArray:
        if cls_name not in self._class_xarray: # has not yet been loaded
            if cls_name == "background" and "background" not in self.annotated_classes:
                # infer background class
                arrs = []
                for cls_iter in self.parent_data.class_list:
                    cls_arr = self.class_xarray(cls_iter)
                    arrs.append(cls_arr)
                self._class_xarray[cls_name] = xr.ones_like(arrs[-1]) - np.sum(arrs, axis=0) # This assumes 1=present and 0=absent, no unknowns
            elif cls_name not in self.annotated_classes:
                msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
                raise ValueError(msg)
            else:
                full_path = self.crop_dir / self.crop_name / self.label_subgroup  / cls_name / self.scales[cls_name]
                self._class_xarray[cls_name] = read_any_xarray(full_path, name=full_path, use_dask=not self.pre_load)  # type: ignore
        return self._class_xarray[cls_name]

    @property
    def class_ids_xarray(self):
        if self._class_ids_xarray is None: # needs to be loaded
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
        if self.classes == ClassOptions.DATASET:
            cls_idx = self.parent_data.dataset_idx
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
            raw_arr = self.raw_xarray.sel(spatial_slice).squeeze().astype("float32")
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
        if self.classes is not None:
            if isinstance(res, tuple):
                res = (*res, cls_idx)
            else:
                res = (res, cls_idx)
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
        # label_representation_in: LabelRepresentation = LabelRepresentation.BINARY,
        # label_representation_out: LabelRepresentation = LabelRepresentation.BINARY,
    ):
        self.zarr_path = zarr_path
        self.zarr_file = zarr.open(self.zarr_path, "r")
        self.digits = len(next(iter(self.zarr_file.keys())))

        for k in self.zarr_file.keys():
            if self.digits != len(k):
                msg = f"Keys of {self.zarr_path} do not have same length ({self.digits})"
                raise ValueError(msg)
        self.raw_channel = raw_channel
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
