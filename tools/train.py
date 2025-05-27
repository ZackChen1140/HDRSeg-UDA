import sys
import os
import gc
import pandas as pd
from PIL import Image
from datetime import datetime

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import evaluate
from torch_optimizer import Lookahead

from transformers import SegformerConfig

from engine.segformer import get_model
from engine.dataloader import get_dataset, RareCategoryManager, InfiniteDataloader
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from engine.validator import Validator
from configs.config import TrainingConfig


def main(cfg: TrainingConfig, exp_name: str, checkpoint: str, log_dir: str):
    currentTime = datetime.now().strftime("%Y%m%d%H%M%S") if log_dir is None else log_dir[-14:]
    tb_dir = f"logs/{exp_name}_{currentTime}" if log_dir is None else f"logs/{log_dir}"

    categories = Category.load(cfg.category_csv)
    writer = SummaryWriter(tb_dir)

    palette = np.array(
        [[cat.r, cat.g, cat.b] for cat in categories],
        dtype=np.uint8
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    set_seed(cfg.seed)

    rcm = RareCategoryManager(categories, cfg.rcs_path, cfg.rcs_temperature) if cfg.rcs_path != "" else None

    train_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        *[transform.ContrastStretch(
                max_intensity=cfg.max_intensity,
                function_name=cfg.contrast_stretch[idx],
                parameter=cfg.img_proc_params[idx]
            ) for idx in range(0, len(cfg.contrast_stretch))
        ],
        transform.ToTensor(),
        transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size),
        transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transform.RandomGaussian(kernel_size=5),
        transform.Normalize(),
    ]
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        *[transform.ContrastStretch(
                max_intensity=cfg.max_intensity,
                function_name=cfg.contrast_stretch[idx],
                parameter=cfg.img_proc_params[idx]
            ) for idx in range(0, len(cfg.contrast_stretch))
        ],
        transform.ToTensor(),
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]
    
    train_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.train_images_root, ann_dir=cfg.train_labels_root, rcm=rcm, transforms=train_transforms)
    val_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.val_images_root, ann_dir=cfg.val_labels_root, rcm=None, transforms=val_transforms)

    train_dataloader = InfiniteDataloader(
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=cfg.pin_memory
    )

    print(f'{cfg.dataset} Dataset Size: {train_dataset.__len__()}')
    print(f'{cfg.dataset} Validation Dataset Size: {val_dataset.__len__()}')
    assert train_dataset.__len__() > 0
    assert val_dataset.__len__() > 0

    metric = evaluate.load("mean_iou", keep_in_memory=True)

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = len(categories)
    
    model = get_model(cfg.model, segconfig)

    base_optimizer = optim.AdamW(
        [
            {'name': 'backbone', 'params': model.encoder.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
            {'name': 'head', 'params': model.decoder.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
        ]
    )
    optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)

    validator = Validator(val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer.optimizer, 1e-4, 1, 1500)
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer.optimizer, cfg.max_iters - 1500, 1)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer.optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[1500]
    )

    # focal_loss = FocalLoss(gamma=2.0, reduction='mean', ignore_index=255)

    scaler = torch.GradScaler(device=device_name)

    model.to(device)
    torch.compile(model)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        last_best = ckpt['best_miou']
        last_iteration = ckpt['iteration']
        optimizer.optimizer.param_groups[0]["lr"] = scheduler._last_lr[0]
        optimizer.optimizer.param_groups[1]["lr"] = scheduler._last_lr[1]
    elif cfg.pretrain_path != "":
        ckpt = torch.load(cfg.pretrain_path)
        model.load_state_dict(ckpt['model_state_dict'])

    best_miou = 0.0 if checkpoint is None else last_best
    begin_iter = 0 if checkpoint is None else last_iteration
    batch_loss, batch_miou = 0.0, 0.0
    for iter in range(begin_iter + 1, cfg.max_iters + 1):
        model.train()
        optimizer.zero_grad()

        data = next(train_dataloader)
        imgs = data["imgs"] if "imgs" in data else [data["img"]]
        ann = data["ann"]
        imgs, ann = [im.to(device) for im in imgs], ann.to(device)
        with torch.autocast(device_type=device_name, enabled=cfg.autocast):
            upsampled_logits, loss = model.forward(
                images=imgs,
                label=ann
            )
            # loss = focal_loss.forward(input=upsampled_logits, target=ann)
        pred = upsampled_logits.argmax(dim=1)
        scaler.scale(loss).backward()

        metrics = metric._compute(
            predictions=pred.cpu(),
            references=ann.cpu(),
            num_labels=segconfig.num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        batch_miou += metrics['mean_iou']
        batch_loss += loss.item()
                
        if iter % cfg.train_interval == 0:
            train_loss = batch_loss / cfg.train_interval
            train_miou = batch_miou / cfg.train_interval
            batch_loss, batch_miou = 0.0, 0.0
            
            writer.add_scalar('Loss/Train', train_loss, iter)
            writer.add_scalar('mIoU/Train', train_miou, iter)
            writer.add_scalar('Learning Rate', optimizer.optimizer.param_groups[0]['lr'], iter)

            print(
                f"[Iter {iter}] Training Loss: {train_loss}, "
                f"Mean_iou: {train_miou}, "
                f"Learning Rate: {optimizer.optimizer.param_groups[0]['lr']}"
            )
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if iter % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device_name):
                val_loss, val_miou, _, iou_list = validator.validate()

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_miou': val_miou,
                    'iteration': iter
                }
                torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/latest_model_checkpoint_{currentTime}.pth")
                if val_miou > best_miou:
                    for file in os.listdir(f'logs/{exp_name}_{currentTime}'):
                        if file.startswith(f"best_model_checkpoint_iter"):
                            os.remove(f"logs/{exp_name}_{currentTime}/{file}")
                    torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/best_model_checkpoint_iter{iter}_{currentTime}.pth")
                    best_miou = val_miou
                
                if iter % (cfg.max_iters / 4) == 0:
                    torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/checkpoint_iter_{iter}_{currentTime}.pth")

                writer.add_scalar('Loss/Validation', val_loss, iter)
                writer.add_scalar('mIoU/Validation', val_miou, iter)
                print(f"Validation Loss: {val_loss}, Mean_iou: {val_miou}")
                print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': iou_list}))
            gc.collect()
    writer.close()

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2 or len(sys.argv) == 3
    cfg = TrainingConfig.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    checkpoint = sys.argv[2] if len(sys.argv) == 3 else None
    log_dir = None if checkpoint is None else checkpoint.split('/')[-2]
    main(cfg, exp_name, checkpoint, log_dir)