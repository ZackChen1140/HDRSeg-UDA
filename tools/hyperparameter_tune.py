import sys
import os
import gc
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import evaluate

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from engine.dataloader import CityscapesLDR, InfiniteDataloader
from engine import transform
from engine.misc import set_seed
from configs.config import TuningConfig

from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

def trainable(config, cfg):
    hyperparam = config
    set_seed(cfg.seed)

    train_transforms = [
        transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size),
        transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transform.RandomGaussian(kernel_size=5),
        transform.Normalize(),
    ]
    val_transforms = [
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]

    train_dataset = CityscapesLDR(img_dir=cfg.train_images_root, ann_dir=cfg.train_labels_root, transforms=train_transforms)
    val_dataset = CityscapesLDR(img_dir=cfg.val_images_root, ann_dir=cfg.val_labels_root, transforms=val_transforms)

    train_dataloader = InfiniteDataloader(
        dataset=train_dataset,
        batch_size=int(hyperparam["batch_size"]),
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory
    )

    metric = evaluate.load("mean_iou")

    segconfig = SegformerConfig().from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    segconfig.num_labels = 19
    
    model = SegformerForSemanticSegmentation(segconfig)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.AdamW(
        [
            {'name': 'backbone', 'params': model.segformer.parameters(), 'lr': hyperparam["backbone_lr"], 'weight_decay': hyperparam["weight_decay"]},
            {'name': 'head', 'params': model.decode_head.parameters(), 'lr': hyperparam["head_lr"], 'weight_decay': hyperparam["weight_decay"]},
        ]
    )

    scalar = torch.GradScaler(device=device)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            last_iteration = checkpoint_state["iteration"]

    begin_iter = 0 if checkpoint is None else last_iteration
    for iter in range(begin_iter + 1, cfg.max_iters + 1):
        model.train()

        inputs, labels = next(train_dataloader)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=inputs, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        
        if iter % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_steps = 0
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(pixel_values=inputs, labels=labels)
                    loss, logits = outputs.loss, outputs.logits
                    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)
                    metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

                    val_loss += loss.item()
                    val_steps += 1

                metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=labels.cpu(),
                    num_labels=segconfig.num_labels,
                    ignore_index=255,
                    reduce_labels=False,
                )

                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iter,
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {"loss": val_loss / val_steps, "mIoU": metrics['mean_iou'], "mAcc": metrics['mean_accuracy']},
                        checkpoint=checkpoint,
                    )
            metric = None
            gc.collect()
            for file in os.listdir('/home/rvl/.cache/huggingface/metrics/mean_io_u/default'):
                os.remove(f"/home/rvl/.cache/huggingface/metrics/mean_io_u/default/{file}")
            if 'evaluate' in sys.modules:
                del sys.modules['evaluate']
            metric = evaluate.load("mean_iou")

def main(cfg: TuningConfig, exp_name: str, log_dir: str):
    currentTime = datetime.now().strftime("%Y%m%d%H%M%S") if log_dir is None else log_dir[-14:]
    tb_dir = f"logs/{exp_name}_{currentTime}" if log_dir is None else f"logs/{log_dir}"
    hyperparameters = {
        "batch_size": tune.choice(cfg.train_batch_size),
        "backbone_lr": tune.loguniform(cfg.backbone_lr[0], cfg.backbone_lr[1]),
        "head_lr": tune.loguniform(cfg.head_lr[0], cfg.head_lr[1]),
        "weight_decay": tune.loguniform(cfg.weight_decay[0], cfg.weight_decay[1]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=cfg.max_iters / cfg.val_interval,
        grace_period=cfg.grace_period,
        reduction_factor=2,
    )
    result = tune.run(
        partial(trainable, cfg=cfg),
        resources_per_trial={"cpu": 10, "gpu": cfg.gpus_per_trial},
        config=hyperparameters,
        num_samples=cfg.num_samples,
        scheduler=scheduler,
        name=exp_name,
        local_dir=tb_dir
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation mIoU: {best_trial.last_result['mIoU']}")
    print(f"Best trial final validation mAcc: {best_trial.last_result['mAcc']}")

if __name__ == "__main__":
    assert len(sys.argv) == 2 or len(sys.argv) == 3
    cfg = TuningConfig.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    log_dir = sys.argv[2] if len(sys.argv) == 3 else None

    main(cfg, exp_name, log_dir)