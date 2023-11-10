import os
from functools import partial
from pathlib import Path

import numpy as np
import zarr
import datatree
from fibsem_tools import read, read_xarray
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T

from denoising_diffusion_pytorch.convenience import exists
import logging


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


class CellMapDatsets3Das2D(ConcatDataset):
    def __init__(
        self,
        data_paths,
        class_list,
        patch_size,
        scale,
        augment_horizontal_flip=True,
        augment_vertical_flip=True,
        annotation_paths=None,
        crop_lists=None,
        raw_datasets=None,
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
                    patch_size,
                    scale,
                    augment_horizontal_flip=augment_horizontal_flip,
                    augment_vertical_flip=augment_vertical_flip,
                    annotation_path=ap,
                    crop_list=cl,
                    raw_dataset=rd,
                )
            )
        super().__init__(cellmap_datasets)


class CellMapDataset3Das2D(ConcatDataset):
    def __init__(
        self,
        data_path,
        class_list,
        patch_size,
        scale,
        augment_horizontal_flip=True,
        augment_vertical_flip=True,
        annotation_path=None,
        crop_list=None,
        raw_dataset="volumes/raw",
    ):
        self.data_path = data_path
        self.raw_dataset = raw_dataset
        self.scale = scale
        self.class_list = class_list
        self.patch_size = patch_size
        self._raw_scale = None
        self.augment_horizontal_flip = augment_horizontal_flip
        self.augment_vertical_flip = augment_vertical_flip
        if annotation_path is None:
            self.annotation_path = data_path
        else:
            self.annotation_path = annotation_path
        self.crops = self._get_crop_list(crop_list)
        super().__init__(self.crops)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
            ]
        )

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.data_path} at {self.scale} with crops {[c.crop_name for c in self.crops]}"

    def _get_crop_list(self, crop_list=None):
        if crop_list is None:
            sample = read(self.annotation_path)
            crops = []
            for ds in sample:
                ann = read(os.path.join(self.annotation_path, ds))
                if "cellmap" in ann.attrs and "annotation" in ann.attrs["cellmap"]:
                    crop = AnnotationCrop3Das2D(self, self.annotation_path, ds)
                    if all(crop.sizes[dim] >= self.patch_size for dim in ['x', 'y']) and all(
                        crop.is_fully_annotated(class_name) for class_name in self.class_list
                    ):
                        crops.append(crop)
                    else:
                        if all(crop.sizes[dim] >= self.patch_size for dim in ['x', 'y']):
                            msg = f"{crop} has sizes {crop.sizes}, which is too small for patch size {self.patch_size}"
                        else:
                            not_incl = []
                            for class_name in self.class_list:
                                if not crop.is_fully_annotated(class_name):
                                    not_incl.append(class_name)
                            msg = f"{not_incl} not annotated in {crop}"
                        logger.debug(msg)
        else:
            crops = [AnnotationCrop3Das2D(self, self.annotation_path, crop_name) for crop_name in crop_list]
        if len(crops) == 0:
            msg = f"List of crops for {self.data_path} with annotations for {self.class_list} at {self.annotation_path} is empty."
            raise ValueError(msg)
        return crops

    def get_raw_xarray(self):
        return read_xarray(os.path.join(self.data_path, self.raw_dataset, self.raw_scale))

    @property
    def raw_scale(self):
        if self._raw_scale is None:
            arr_path = os.path.join(self.data_path, self.raw_dataset)
            msarr = read_xarray(arr_path)
            if isinstance(msarr, datatree.DataTree):
                scale_to_voxelsize = dict()
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

    def __getitem__(self, idx):
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
    def __init__(self, parent_data, annotation_path, crop_name):
        self.parent_data = parent_data
        self.annotation_path = annotation_path
        self.crop_name = crop_name
        self.crop = read(os.path.join(self.annotation_path, self.crop_name))
        if "cellmap" not in self.crop.attrs or "annotation" not in self.crop.attrs["cellmap"]:
            msg = f"Crop {crop_name} at {annotation_path} is not a cellmap annotation crop."
            raise ValueError(msg)
        self._scales = None
        self._sizes = None
        self._size = None
        self._coords = None
        self.annotated_classes = self.crop.attrs["cellmap"]["annotation"]["class_names"]
        self.class_list = list(set(self.annotated_classes).intersection(set(self.parent_data.class_list)))

    @property
    def scales(self):
        if self._scales is None:
            self._scales = dict()
            for class_name in self.annotated_classes:
                self._scales[class_name] = self._infer_scale_level(class_name)
        return self._scales

    @property
    def sizes(self):
        if self._sizes is None:
            self._sizes = self.get_xarray_attr('sizes')
        return self._sizes

    @property
    def size(self):
        if self._size is None:
            self._size = self.get_xarray_attr('size')
        return self._size

    @property
    def coords(self):
        if self._coords is None:
            self._coords = self.get_xarray_attr('coords')
        return self._coords

    def get_xarray_attr(self, attr):
        ref_attr = None
        for class_name in self.class_list:
            curr_attr = getattr(self.get_class_xarray(class_name), attr)
            if ref_attr is None:
                ref_attr = curr_attr
            elif curr_attr != ref_attr:
                msg = f"Crop {self} has arrays with different values for {attr} for requested scale {self.parent_data.scale}. Found (at least) {curr_attr}, {ref_attr}."
                raise ValueError(msg)
        return ref_attr

    def __repr__(self):
        return f"{self.__class__.__name__} {self.crop_name} at {self.annotation_path} from {self.parent_data.__class__.__name__} at {self.parent_data.data_path}"

    def _infer_scale_level(self, cls_name):
        if cls_name not in self.crop:
            msg = f"{cls_name} not found in {self}"
            raise ValueError(msg)
        arr_path = os.path.join(self.annotation_path, self.crop_name, cls_name)
        cls_msarr = read_xarray(arr_path)
        scale_to_voxelsize = dict()
        for name, dtarr in cls_msarr.children.items():
            arr = dtarr.data
            voxel_size = {ax: arr[ax].values[1] - arr[ax].values[0] for ax in arr.dims}
            if voxel_size == self.parent_data.scale:
                return name
            scale_to_voxelsize[cls_name] = voxel_size
        msg = f"{arr_path} does not contain array with voxel_size {self.parent_data.scale}. Available scale levels are: {scale_to_voxelsize}"
        raise ValueError(msg)

    def get_classes(self):
        return self.crop.attrs["cellmap"]["annotation"]["class_names"]

    def get_class_array(self, cls_name):
        if cls_name not in self.annotated_classes:
            msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
            raise ValueError(msg)
        return self.crop[os.path.join(cls_name, self.scales[cls_name])]

    def get_class_xarray(self, cls_name):
        if cls_name not in self.annotated_classes:
            msg = f"{cls_name} is not part of the annotated classes {self.annotated_classes}."
            raise ValueError(msg)
        return read_xarray(os.path.join(self.annotation_path, self.crop_name, cls_name, self.scales[cls_name]))

    def get_counts(self, cls_name):
        return self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["complement_counts"]

    def get_possibilities(self, cls_name):
        return set(self.get_class_array(cls_name).attrs["cellmap"]["annotation"]["annotation_type"]["encoding"].keys())

    def get_present_count(self, cls_name):
        counts = self.get_counts(cls_name)
        possiblities = self.get_possibilities(cls_name)
        possiblities.remove("present")
        not_present_sum = 0
        for possibility in possiblities:
            if possibility in counts:
                not_present_sum += counts[possibility]
        return self.size - not_present_sum

    def has_present(self, cls_name):
        return self.get_present_count(cls_name) > 0

    def is_fully_annotated(self, cls_name):
        if cls_name not in self.annotated_classes:
            return False
        counts = self.get_counts(cls_name)
        if "unknown" not in counts:
            return True
        else:
            return counts["unknown"] == 0

    def __len__(self):
        return self.sizes['z']

    def __getitem__(self, idx):
        vox_slice = dict()
        vox_slice["z"] = idx
        x_start = np.random.randint(0, self.sizes["x"] - self.parent_data.patch_size + 1)
        y_start = np.random.randint(0, self.sizes["y"] - self.parent_data.patch_size + 1)
        vox_slice["x"] = slice(x_start, x_start + self.parent_data.patch_size)
        vox_slice["y"] = slice(y_start, y_start + self.parent_data.patch_size)
        arrs = []
        for cls_name in self.parent_data.class_list:
            cls_arr = self.get_class_xarray(cls_name).isel(vox_slice)
            arrs.append(cls_arr.copy())
        spatial_slice = {
            dim: slice(
                int(cls_arr.coords[dim][0]) - self.parent_data.scale[dim] / 2,
                int(cls_arr.coords[dim][-1]) + self.parent_data.scale[dim] / 2,
            )
            for dim in "xy"
        }
        spatial_slice["z"] = slice(
            int(cls_arr.coords["z"]) - self.parent_data.scale["z"] / 2,
            int(cls_arr.coords["z"]) + self.parent_data.scale["z"] / 2,
        )
        raw_arr = self.parent_data.get_raw_xarray().sel(spatial_slice).squeeze().copy() / 255.0
        arrs.append(raw_arr)
        patch = np.stack(tuple(arr.values.astype(np.float32) for arr in arrs), axis=-1)
        return patch


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
