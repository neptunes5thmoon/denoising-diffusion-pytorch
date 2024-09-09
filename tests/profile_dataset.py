import cProfile
import os
import time
import warnings
from urllib.parse import urlparse, urlunparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml

from denoising_diffusion_pytorch import CellMapDatasets3Das2D, GaussianDiffusion, Trainer, Unet

warnings.filterwarnings("ignore", module="pydantic_ome_ngff")  # line104


def plot_example(arr):
    arr = arr.cpu()
    if arr.ndim == 4:
        arr = arr[np.random.randint(arr.shape[0])]
    # fig, axs= plt.subplots(2,1)
    fig, axs = plt.subplots(4, 8)
    for ch, ax in enumerate(axs.flatten()):
        a = arr[ch - 1, ...]
        print(a.min(), a.max())
        ax.imshow(a, vmin=0, vmax=1, cmap="Greys_r")
    fig.show()


def run(iterations, dataloader=False, all_data=False, plot=True):
    # Load configuration from YAML file
    data_args = dict()
    data_args["image_size"] = 96
    if all_data:
        data_args["data_paths"] = [
            "/nrs/cellmap/data/jrc_hela-2/jrc_hela-2.n5",
            "/nrs/cellmap/data/jrc_hela-3/jrc_hela-3.n5",
            "/nrs/cellmap/data/jrc_macrophage-2/jrc_macrophage-2.n5",
            "/nrs/cellmap/data/jrc_sum159-1/jrc_sum159-1.n5",
            "/nrs/cellmap/data/jrc_jurkat-1/jrc_jurkat-1.n5",
        ]
        data_args["crop_lists"] = [
            [
                "crop1",
                "crop113",
                "crop13",
                "crop14",
                "crop15",
                "crop155",
                "crop16",
                "crop18",
                "crop19",
                "crop23",
                "crop28",
                "crop3",
                "crop4",
                "crop6",
                "crop7",
                "crop8",
                "crop9",
            ],
            ["crop101", "crop111", "crop181", "crop27", "crop33", "crop34", "crop50", "crop51"],
            ["crop109", "crop110", "crop31", "crop32", "crop39", "crop40", "crop42", "crop48", "crop49"],
            ["crop20", "crop21", "crop22", "crop80"],
            [
                "crop107",
                "crop112",
                "crop126",
                "crop180",
                "crop182",
                "crop35",
                "crop36",
                "crop37",
                "crop38",
                "crop43",
                "crop47",
            ],
        ]
        data_args["annotation_paths"] = [
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_hela-2/groundtruth.zarr",
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_hela-3/groundtruth.zarr",
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_macrophage-2/groundtruth.zarr",
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_sum159-1/groundtruth.zarr",
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_jurkat-1/groundtruth.zarr",
        ]
        data_args["raw_datasets"] = [None, None, None, None, None]
    else:
        data_args["data_paths"] = ["/nrs/cellmap/data/jrc_sum159-1/jrc_sum159-1.n5"]
        data_args["crop_lists"] = [
            ["crop20", "crop21", "crop22", "crop80"],
        ]
        data_args["annotation_paths"] = [
            "/home/heinrichl@hhmi.org/data/cellmap/reslicing/jrc_sum159-1/groundtruth.zarr",
        ]
        data_args["raw_datasets"] = [
            None,
        ]
    data_args["class_list"] = [
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
    ]
    data_args["scale"] = {"x": 4, "y": 4, "z": 4}
    data_args["augment_horizontal_flip"] = True
    data_args["augment_vertical_flip"] = True
    data_args["dask_workers"] = 1
    data_args["pre_load"] = True
    dataset = CellMapDatasets3Das2D(**data_args)
    trainer_args = {
        "train_batch_size": 128,
        "train_lr": 8e-5,
        "train_num_steps": 700000,
        "gradient_accumulate_every": 1,
        "ema_decay": 0.995,
        "amp": False,
        "calculate_fid": False,
        "dataloader_nworkers": 78,
        "persistent_workers": True,
        "prefetch_factor": 1,
    }
    if dataloader:
        architecture_args = {"dim": 64, "channels": 32, "dim_mults": [1, 2, 4, 8]}
        architecture = Unet(**architecture_args)
        diffusion_args = {
            "image_size": data_args["image_size"],
            "timesteps": 1000,
            "sampling_timesteps": 250,
            "auto_normalize": False,
            "channel_weights": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                31,
            ],
        }
        diffusion = GaussianDiffusion(architecture, **diffusion_args)

        trainer = Trainer(diffusion, dataset, results_folder="results", **trainer_args)
    times = []
    print("Start measurement...")
    if dataloader:
        for t in range(iterations):
            start = time.time()
            a = next(trainer.dl)
            if plot:
                plot_example(a)
            interval = time.time() - start
            print(f"interval # {t}: {interval}")
            times.append(interval)
    else:
        print()
        print(len(dataset))
        print()
        a = np.random.randint(0, len(dataset), size=(iterations, trainer_args["train_batch_size"]))
        for t in range(a.shape[0]):
            start = time.time()
            for b in range(a.shape[1]):
                x = dataset[a[t, b]]
                if plot:
                    plot_example(x)
            interval = time.time() - start
            print(f"interval # {t}: {interval}")
            times.append(interval)
    print(f"Average: {np.mean(times)}", flush=True)
    if plot:
        input("Press enter to continue")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("iterations", type=int)
    parser.add_argument("--dataloader", action="store_true")
    parser.add_argument("--all_data", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.iterations, dataloader=args.dataloader, all_data=args.all_data, plot=args.plot)
