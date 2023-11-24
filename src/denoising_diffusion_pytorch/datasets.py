from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import zarr
import datatree
from fibsem_tools import read, read_xarray
from fibsem_tools.io.util import ArrayLike, GroupLike
from PIL import Image
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T

from denoising_diffusion_pytorch.convenience import exists
import logging
import xarray as xr
from datatree import DataTree
import dask
from typing import Any, Mapping, Union, Sequence

logger = logging.getLogger(__name__)


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
        raw_datasets: Sequence[str] | None = None,
        dask_workers: int = 0,
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
        allow_single_class_crops: Sequence[str | None]
        | None = None,  # only has an effect if crop_list is not specified
        annotation_path: str | None = None,
        crop_list: Sequence[str] | None = None,
        raw_dataset: str | None = "volumes/raw",
        dask_workers=0,
    ) -> None:
        self.data_path = data_path
        self.raw_dataset = raw_dataset
        self.scale = scale
        self.class_list = class_list
        self.image_size = image_size
        self._raw_scale: str | None = None
        self.augment_horizontal_flip = augment_horizontal_flip
        self.augment_vertical_flip = augment_vertical_flip
        self.dask_workers = dask_workers
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
                T.ToTensor(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
            ]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.data_path} at {self.scale} with crops {[c.crop_name for c in self.crops]}"

    def _get_crop_list(self, crop_list: Sequence[str] | None = None) -> list[AnnotationCrop3Das2D]:
        filtering = self.allow_single_class_crops != set(self.class_list).union({None})
        if crop_list is None:
            sample: GroupLike = read(self.annotation_path)
            crops = []
            for ds in sample:
                ann = read(os.path.join(self.annotation_path, ds))
                if "cellmap" in ann.attrs and "annotation" in ann.attrs["cellmap"]:
                    crop = AnnotationCrop3Das2D(self, self.annotation_path, ds, dask_workers=self.dask_workers)
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
                AnnotationCrop3Das2D(self, self.annotation_path, crop_name, dask_workers=self.dask_workers)
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
    def raw_scale(self) -> str:
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
    def __init__(self, parent_data: CellMapDataset3Das2D, annotation_path: str, crop_name: str, dask_workers: int = 0):
        self.parent_data = parent_data
        self.annotation_path = annotation_path
        self.crop_name = crop_name
        self.crop: ArrayLike = read(os.path.join(self.annotation_path, self.crop_name))  # type: ignore
        if "cellmap" not in self.crop["labels"].attrs or "annotation" not in self.crop["labels"].attrs["cellmap"]:
            msg = f"Crop {crop_name} at {annotation_path} is not a cellmap annotation crop."
            raise ValueError(msg)
        self._scales: dict[str, str] | None = None
        self._sizes: None | Mapping[str, int] = None
        self._size: None | int = None
        self._coords: None | xr.Coordinates = None
        self.annotated_classes = self.crop["labels"].attrs["cellmap"]["annotation"]["class_names"]
        self.class_list = list(set(self.annotated_classes).intersection(set(self.parent_data.class_list)))
        self._class_xarray: dict[str, xr.DataArray] = dict()
        self._raw_xarray = None
        self.dask_workers = dask_workers

    @property
    def raw_xarray(self):
        if self._raw_xarray is None:
            if "raw" in self.crop:
                try:
                    mslvl = self._infer_scale_level("raw")
                    self._raw_xarray = read_xarray(os.path.join(self.annotation_path, self.crop_name, "raw", mslvl))
                except ValueError:
                    if self.parent_data.raw_dataset is None:
                        raise ValueError("bla")
                    self._raw_xarray = self.parent_data.raw_xarray.copy()
            else:
                if self.parent_data.raw_dataset is None:
                    raise ValueError("bla")
                self._raw_xarray = self.parent_data.raw_xarray.copy()
        return self._raw_xarray

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
        return self.crop["labels"].attrs["cellmap"]["annotation"]["class_names"]

    def get_class_array(self, cls_name: str):
        if cls_name not in self.annotated_classes:
            msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
            raise ValueError(msg)
        return self.crop[os.path.join("labels", cls_name, self.scales[cls_name])]

    def class_xarray(self, cls_name: str) -> xr.DataArray:
        if cls_name not in self._class_xarray:
            if cls_name not in self.annotated_classes:
                msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
                raise ValueError(msg)
            self._class_xarray[cls_name] = read_xarray(os.path.join(self.annotation_path, self.crop_name, "labels", cls_name, self.scales[cls_name]))  # type: ignore
        return self._class_xarray[cls_name]

    def get_counts(self, cls_name: str) -> Mapping[str, int]:
        return self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["complement_counts"]

    def get_possibilities(self, cls_name: str) -> set[str]:
        return set(self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["annotation_type"]["encoding"].keys())

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
        return self.sizes["z"]

    def __getitem__(self, idx: int) -> np.ndarray:
        x_start = np.random.randint(0, self.sizes["x"] - self.parent_data.image_size + 1)
        y_start = np.random.randint(0, self.sizes["y"] - self.parent_data.image_size + 1)
        vox_slice = {
            "z": idx,
            "x": slice(x_start, x_start + self.parent_data.image_size),
            "y": slice(y_start, y_start + self.parent_data.image_size),
        }
        arrs = []
        for cls_name in self.parent_data.class_list:
            cls_arr = self.class_xarray(cls_name).isel(vox_slice)
            arrs.append(cls_arr.astype(np.float32))
        spatial_slice = {
            dim: slice(
                int(cls_arr.coords[dim][0])
                - self.parent_data.scale[dim] / 2,  # hack to deal with misalignment/wrong offsets
                int(cls_arr.coords[dim][-1]) + self.parent_data.scale[dim] / 2,
            )
            for dim in "xy"
        }
        spatial_slice["z"] = slice(
            int(cls_arr.coords["z"]) - self.parent_data.scale["z"] / 2,
            int(cls_arr.coords["z"]) + self.parent_data.scale["z"] / 2,
        )
        raw_arr = self.raw_xarray.sel(spatial_slice).squeeze() / 255.0
        arrs.append(raw_arr)
        patch = dask.array.stack(arrs, axis=-1).compute(num_workers=self.dask_workers)

        return patch  # shape (self.parent_data.image_size, self.parent_data.image_size, len(self.parent_data.class_list)+1)


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
