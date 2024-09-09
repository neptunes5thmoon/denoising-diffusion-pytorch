import math
import os
from enum import Enum
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

import mlflow
import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.convenience import cycle, divisible_by, exists
from denoising_diffusion_pytorch.version import __version__


class SegmentationMetrics(Enum):
    CROSS_ENTROPY = partial(nn.CrossEntropyLoss())

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class SegmentationActivation(Enum):
    SOFTMAX = partial(nn.Softmax(dim=1))

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


SegmentationMetricsNames = Literal[tuple(e.name for e in SegmentationMetrics)]
SegmentationActivationNames = Literal[tuple(e.name for e in SegmentationActivation)]


class BaselineSegmentationInference(nn.Module):
    def __init__(self, model, activation):
        super().__init__()
        self.model = model
        self.activation = SegmentationActivation[activation]

    def forward(self, img, *args, **kwargs):
        model_out = self.model(img)
        return self.activation(model_out, *args, **kwargs)


class BaselineSegmentation(nn.Module):
    def __init__(self, model, image_size, loss_fn="CROSS_ENTROPY", activation=None):
        super().__init__()
        self.model = model
        if activation is None:
            self.inference_model = self.model
        else:
            self.inference_model = BaselineSegmentationInference(self.model, activation)
        self.image_size = image_size
        self.loss_fn = SegmentationMetrics[loss_fn]

    def forward(self, img, target, *args, **kwargs):
        # todo normalize?
        model_out = self.model(img)
        return self.loss_fn(model_out, torch.squeeze(target), *args, **kwargs)


class BaselineSegmentationPredictor:
    def __init__(
        self,
        segmentation_model,
        dataset,
        exporter,
        *,
        batch_size=16,
        criteria=None,
        results_folder="./results",
        dataloader_nworkers=cpu_count(),
        amp=False,
        split_batches=True,
        mixed_precision_type="fp16",
        milestone=None,
        include_input=True,
    ):
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )
        self.exporter = exporter
        self.model = segmentation_model
        self.batch_size = batch_size
        self.ds = dataset
        self.include_input = include_input
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model = self.accelerator.prepare(self.model)
        dl = DataLoader(self.ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=dataloader_nworkers)
        self.dl = self.accelerator.prepare(dl)
        self.criteria = dict()
        if criteria is not None:
            for criterion_name in criteria:
                self.criteria[criterion_name] = SegmentationMetrics[criterion_name]
        milestones = [ckpt.split("_")[1] for ckpt in os.listdir(self.results_folder) if ckpt.startswith("ckpt_")]
        self.milestone_digits = len(milestones[0])
        if milestone is None:
            self.load_last()
        else:
            self.load(milestone)

    @property
    def device(self):
        return self.accelerator.device

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(
                self.results_folder
                / f"ckpt_{milestone:0{self.milestone_digits}d}"
                / f"model_{milestone:0{self.milestone_digits}d}.pt"
            ),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.milestone = data["step"]
        # if self.accelerator.is_main_process:
        # self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def load_last(self):
        milestones = [int(ckpt.split("_")[1]) for ckpt in os.listdir(self.results_folder)]
        if len(milestones) > 0:
            self.load(max(milestones))

    def inference(self):
        accelerator = self.accelerator
        if accelerator.is_main_process:
            self.model.inference_model.eval()
            with torch.inference_mode():
                for data in self.dl:
                    prediction = self.model.inference_model(data)
                    all_predictions = accelerator.gather(prediction)
                    if self.include_input:
                        all_data = accelerator.gather(data)
                        torch_arr = torch.cat([all_data, all_predictions], axis=1)
                    else:
                        torch_arr = all_predictions
                    self.exporter.save_sample(
                        str(self.results_folder / f"ckpt_{self.milestone:0{self.milestone_digits}d}"), torch_arr
                    )

        return all_predictions


class BaselineSegmentationTrainer:
    def __init__(
        self,
        segmentation_model,
        dataset,
        loader_exporter,
        prediction_exporter,
        *,
        train_batch_size=16,
        validation_dataset=None,
        validation_criteria=None,
        validation_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        dataloader_nworkers=cpu_count(),
        persistent_workers=True,
        prefetch_factor=2,
        shuffle_dataloader=True,
        repeat_data=True,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )
        # saver
        self.loader_exporter = loader_exporter
        self.prediction_exporter = prediction_exporter

        # model

        self.model = segmentation_model
        # is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        # self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        if not repeat_data and self.train_num_steps > len(dataset):
            self.train_num_steps = len(dataset)
        self.image_size = segmentation_model.image_size

        # self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = dataset
        self.validation_ds = validation_dataset
        assert (
            len(self.ds) >= 100
        ), "you should have at least 100 images in your folder. at least 10k images recommended"

        # optimizer
        self.opt = Adam(segmentation_model.parameters(), lr=train_lr, betas=adam_betas)
        self.max_grad_norm = max_grad_norm
        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        #     self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        #     self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        total_milestones = math.ceil(self.train_num_steps / self.save_and_sample_every)
        self.milestone_digits = len(str(total_milestones))

        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=shuffle_dataloader,
            pin_memory=True,
            num_workers=dataloader_nworkers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

        dl = self.accelerator.prepare(dl)

        self.dl = cycle(dl)
        if self.validation_ds is not None:
            val_dl = DataLoader(
                self.validation_ds,
                batch_size=validation_batch_size,
                shuffle=False,
                num_workers=max(1, int(dataloader_nworkers / 10)),
                persistent_workers=False,
            )
            self.val_dl = self.accelerator.prepare(val_dl)
        else:
            self.val_dl = None

        self.criteria = dict()
        if validation_criteria is not None:
            for criterion_name in validation_criteria:
                self.criteria[criterion_name] = SegmentationMetrics[criterion_name]
        self.step = 0

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            # "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }
        ckpt_dir = self.results_folder / f"ckpt_{milestone:0{self.milestone_digits}d}"
        ckpt_dir.mkdir(exist_ok=True)
        model_path = str(ckpt_dir / f"model_{milestone:0{self.milestone_digits}d}.pt")
        torch.save(data, model_path)

    def load_last(self):
        milestones = [int(ckpt.split("_")[1]) for ckpt in os.listdir(self.results_folder)]
        if len(milestones) > 0:
            self.load(max(milestones))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(
                self.results_folder
                / f"ckpt_{milestone:0{self.milestone_digits}d}"
                / f"model_{milestone:0{self.milestone_digits}d}.pt"
            ),
            map_location=device,
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        # if self.accelerator.is_main_process:
        # self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.load_last()
        if self.val_dl is not None:
            for data, target in self.val_dl:
                all_data = accelerator.gather(data)
                all_target = accelerator.gather(target)
                self.loader_exporter.save_sample(
                    str(self.results_folder / "reference"), torch.cat([all_data, all_target], dim=1)
                )
        self.train_num_steps = self.train_num_steps + self.step
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                for _ in range(self.gradient_accumulate_every):
                    data, target = next(self.dl)
                    # all_data = accelerator.gather(data)
                    # all_target = accelerator.gather(target)
                    # self.loader_exporter.save_sample(str(self.results_folder / f"batch{self.step:0{self.milestone_digits}d}"),
                    #                                  torch.cat([all_data, all_target], dim=1))
                    with self.accelerator.autocast():
                        loss = self.model(data, target)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                mlflow.log_metric("loss", total_loss, step=self.step)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    # self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        # self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        if self.val_dl is not None:
                            self.model.inference_model.eval()
                            with torch.inference_mode():
                                criterion_values = {criterion_name: 0.0 for criterion_name in self.criteria.keys()}
                                for data, target in self.val_dl:
                                    prediction = self.model.inference_model(data)
                                    all_predictions = accelerator.gather(prediction)
                                    all_targets = accelerator.gather(target)
                                    for criterion_name, criterion_function in self.criteria.items():
                                        criterion_values[criterion_name] += criterion_function(
                                            all_predictions, torch.squeeze(all_targets)
                                        )
                                    self.prediction_exporter.save_sample(
                                        str(self.results_folder / f"ckpt_{milestone:0{self.milestone_digits}d}"),
                                        all_predictions,
                                    )
                                for criterion_name, criterion_value in criterion_values.items():
                                    mlflow.log_metric(
                                        criterion_name, criterion_value / len(self.val_dl), step=self.step
                                    )
                            self.model.model.train()
                pbar.update(1)

        accelerator.print("training complete")
