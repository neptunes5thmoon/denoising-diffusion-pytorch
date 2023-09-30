from functools import partial
from pathlib import Path

import zarr

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
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
        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if exists(convert_image_to)
            else nn.Identity()
        )

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
        load_to_ram = True
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
                #RandomNonEmptyCrop(image_size, padding=0),
                T.RandomCrop(image_size, padding=0),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                #T.ToTensor(),
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
        img = arr[...,z].transpose((1,2,0))
        return self.transform(img)