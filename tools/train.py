import sys
import os
import gc
import pandas as pd
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import evaluate

from transformers import SegformerConfig

from engine.segformer import MultiPathSegformer, DualPathSegformer
from engine.dataloader import CityscapesHDR_NLCS2, CityscapesHDR_GC3, InfiniteDataloader, RareCategoryManager
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from configs.config import TrainingConfig


def main(cfg: TrainingConfig, exp_name: str, checkpoint: str, log_dir: str):
    currentTime = datetime.now().strftime("%Y%m%d%H%M%S") if log_dir is None else log_dir[-14:]
    tb_dir = f"logs/{exp_name}_{currentTime}" if log_dir is None else f"logs/{log_dir}"

    # dataset_info = pd.read_csv(cfg.category_csv)
    # categories = dataset_info['name'].to_list()
    categories = Category.load(cfg.category_csv)
    writer = SummaryWriter(tb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    rcm = RareCategoryManager(categories, cfg.rcs_path, cfg.rcs_temperature) if cfg.rcs_path != "" else None

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

    train_dataset = CityscapesHDR_NLCS2(img_dir=cfg.train_images_root, ann_dir=cfg.train_labels_root, rcm=rcm, transforms=train_transforms, parameters=cfg.img_proc_params)
    val_dataset = CityscapesHDR_NLCS2(img_dir=cfg.val_images_root, ann_dir=cfg.val_labels_root, rcm=None, transforms=val_transforms, parameters=cfg.img_proc_params)

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
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=cfg.pin_memory
    )

    metric = evaluate.load("mean_iou", keep_in_memory=True)

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = 19
    
    model = DualPathSegformer(segconfig)

    optimizer = optim.AdamW(
        [
            {'name': 'backbone', 'params': model.encoder.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
            {'name': 'head', 'params': model.decoder.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
        ]
    )

    scalar = torch.GradScaler(device=device)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        last_best = ckpt['best_miou']
        last_iteration = ckpt['iteration']
    else:
        model.to(device)

    best_miou = 0 if checkpoint is None else last_best
    begin_iter = 0 if checkpoint is None else last_iteration
    batch_loss, batch_miou, batch_acc = 0, 0, 0
    for iter in range(begin_iter + 1, cfg.max_iters + 1):
        model.train()
        optimizer.zero_grad()

        inputs, labels = next(train_dataloader)
        inputs, labels = [im.to(device) for im in inputs], labels.to(device)

        predicted, loss = model.forward(
            inputs=inputs,
            labels=labels
        )

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        with torch.no_grad():
            metrics = metric._compute(
                predictions=predicted.cpu(),
                references=labels.cpu(),
                num_labels=segconfig.num_labels,
                ignore_index=255,
                reduce_labels=False,
            )
            batch_miou += metrics['mean_iou']
            batch_acc += metrics['mean_accuracy']
            batch_loss += loss.item()

        if iter % cfg.train_interval == 0:
            train_loss = batch_loss / cfg.train_interval
            train_miou = batch_miou / cfg.train_interval
            train_acc = batch_acc / cfg.train_interval
            batch_loss, batch_miou, batch_acc = 0, 0, 0
            
            writer.add_scalar('Loss/Train', train_loss, iter)
            writer.add_scalar('mIoU/Train', train_miou, iter)
            writer.add_scalar('mAcc/Train', train_acc, iter)
            print(f"[Iter {iter}] Training Loss: {train_loss}, Mean_iou: {train_miou}, Mean accuracy: {train_acc}")
        
        if iter % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_batch = 0.0
                iou_list = None
                batch_list = None
                for inputs, labels in val_dataloader:
                    inputs, labels = [im.to(device) for im in inputs], labels.to(device)

                    predicted, loss = model.forward(
                        inputs=inputs,
                        labels=labels
                    )
                    metrics = metric._compute(
                        predictions=predicted.cpu(),
                        references=labels.cpu(),
                        num_labels=segconfig.num_labels,
                        ignore_index=255,
                        reduce_labels=False,
                    )
                    val_batch += 1
                    batch_loss += loss.item()
                    batch_miou += metrics['mean_iou']
                    batch_acc += metrics['mean_accuracy']
                    if iou_list == None:
                        iou_list = metrics['per_category_iou'].tolist()
                        batch_list = [0.0 if math.isnan(v) else 1.0 for v in iou_list]
                    else:
                        for idx, iou in enumerate(metrics['per_category_iou'].tolist()):
                            if not math.isnan(iou):
                                iou_list[idx] = iou if math.isnan(iou_list[idx]) else iou_list[idx] + iou
                                batch_list[idx] += 1

                val_loss = batch_loss / val_batch
                val_miou = batch_miou / val_batch
                val_acc = batch_acc / val_batch
                iou_list = [v / b for v, b in zip(iou_list, batch_list)]
                batch_loss, batch_miou, batch_acc = 0, 0, 0

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
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

                writer.add_scalar('Loss/Validation', val_loss, iter)
                writer.add_scalar('mIoU/Validation', val_miou, iter)
                writer.add_scalar('mAcc/Validation', val_acc, iter)
                print(f"Validation Loss: {val_loss}, Mean_iou: {val_miou}, Mean accuracy: {val_acc}")
                print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': iou_list}))
    writer.close()

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2 or len(sys.argv) == 3
    cfg = TrainingConfig.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    checkpoint = sys.argv[2] if len(sys.argv) == 3 else None
    log_dir = None if checkpoint is None else checkpoint.split('/')[-2]
    main(cfg, exp_name, checkpoint, log_dir)