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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import evaluate

from ema_pytorch import EMA
from transformers import SegformerConfig

from engine.segformer import get_model
from engine.dataloader import get_dataset, RareCategoryManager, InfiniteDataloader
from engine.category import Category
from engine.ema import PixelThreshold
from engine import transform
from engine.misc import set_seed
from engine.validator import Validator
from configs.config import TrainingConfig_UDA


def main(cfg: TrainingConfig_UDA, exp_name: str, checkpoint: str, log_dir: str):
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

    source_train_transforms = [
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
    target_train_transforms = [
        transform.LoadImg(),
        *[transform.ContrastStretch(
                max_intensity=cfg.max_intensity,
                function_name=cfg.contrast_stretch[idx],
                parameter=cfg.img_proc_params[idx]
            ) for idx in range(0, len(cfg.contrast_stretch))
        ],
        transform.ToTensor(),
        transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size),
        *[transform.RandomErase(scale=(0.02, 0.04)) for _ in range(cfg.num_masks)],
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
    
    source_train_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.source_train_images_root, ann_dir=cfg.source_train_labels_root, rcm=rcm, transforms=source_train_transforms)
    target_train_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.target_train_images_root, ann_dir=None, rcm=None, transforms=target_train_transforms)
    source_val_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.source_val_images_root, ann_dir=cfg.source_val_labels_root, rcm=None, transforms=val_transforms)
    target_val_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.target_val_images_root, ann_dir=cfg.target_val_labels_root, rcm=None, transforms=val_transforms)

    source_train_dataloader = InfiniteDataloader(
        dataset=source_train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory
    )
    target_train_dataloader = InfiniteDataloader(
        dataset=target_train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=cfg.pin_memory
    )
    source_val_dataloader = DataLoader(
        dataset=source_val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=cfg.pin_memory
    )
    target_val_dataloader = DataLoader(
        dataset=target_val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=cfg.pin_memory
    )

    print(f'{cfg.dataset} Source Training Dataset Size: {source_train_dataset.__len__()}')
    print(f'{cfg.dataset} Target Training Dataset Size: {target_train_dataset.__len__()}')
    print(f'{cfg.dataset} Source Validation Dataset Size: {source_val_dataset.__len__()}')
    print(f'{cfg.dataset} Source Validation Dataset Size: {target_val_dataset.__len__()}')
    assert source_train_dataset.__len__() > 0
    assert target_train_dataset.__len__() > 0
    assert source_val_dataset.__len__() > 0
    assert target_val_dataset.__len__() > 0

    source_class_weight = torch.zeros((len(categories)))
    target_class_weight = torch.zeros((len(categories)))

    metric = evaluate.load("mean_iou", keep_in_memory=True)

    target_criterion = PixelThreshold()

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = len(categories)
    
    model = get_model(cfg.model, segconfig)
    # ema = EMA(model, beta=0.999, update_after_step=-1, update_every=cfg.ema_update_intervals)
    ema = EMA(model, beta=0.999, update_after_step=-1)


    optimizer = optim.AdamW(
        [
            {'name': 'backbone', 'params': model.encoder.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
            {'name': 'head', 'params': model.decoder.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
        ]
    )

    source_validator = Validator(source_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')
    target_validator = Validator(target_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1e-4, 1, 1500)
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, cfg.max_iters - 1500, 1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[1500]
    )

    scaler = torch.GradScaler(device=device_name)

    source_class_weight.to(device)
    target_class_weight.to(device)
    model.to(device)
    ema.to(device)
    ema.ema_model.to(device)
    torch.compile(model)
    torch.compile(ema.ema_model)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        source_last_best = ckpt['source_best_miou']
        target_last_best = ckpt['target_best_miou']
        last_iteration = ckpt['iteration']

    ema_milestone = (cfg.max_iters + len(cfg.ema_update_intervals)) // len(cfg.ema_update_intervals)
    source_best_miou = 0.0 if checkpoint is None else source_last_best
    target_best_miou = 0.0 if checkpoint is None else target_last_best
    begin_iter = 0 if checkpoint is None else last_iteration
    source_batch_loss, source_batch_miou, target_batch_loss, target_batch_num = 0.0, 0.0, 0.0, 0.0
    for iter in range(begin_iter + 1, cfg.max_iters + 1):
        
        if iter % ema_milestone == 0:
            ema.update_every = cfg.ema_update_intervals[iter // ema_milestone]
        ema.update()

        model.train()
        optimizer.zero_grad()

        source_data = next(source_train_dataloader)
        source_imgs = source_data["imgs"] if "imgs" in source_data else [source_data["img"]]
        source_ann = source_data["ann"]
        source_imgs, source_ann = [im.to(device) for im in source_imgs], source_ann.to(device)
        with torch.autocast(device_type=device_name, enabled=cfg.autocast):
            upsampled_logits, source_loss = model.forward(
                images=source_imgs,
                label=source_ann
            )
        source_pred = upsampled_logits.argmax(dim=1)

        scaler.scale(source_loss).backward()

        metrics = metric._compute(
            predictions=source_pred.cpu(),
            references=source_ann.cpu(),
            num_labels=segconfig.num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        source_batch_miou += metrics['mean_iou']
        source_batch_loss += source_loss.item()

        if iter % ema.update_every == 0:
            target_data = next(target_train_dataloader)
            target_imgs = target_data["imgs"] if "imgs" in target_data else [target_data["img"]]
            target_imgs = [im.to(device) for im in target_imgs]
            if cfg.num_masks > 0:
                erased_target_imgs = target_data["erased imgs"] if "erased imgs" in target_data else [target_data["erased img"]]
                erased_target_imgs = [im.to(device) for im in erased_target_imgs]

            # Pseudo labeling for SEMA and compute class-conditional pixel weights for CCDD
            with torch.no_grad(), torch.autocast(device_type=device_name, enabled=cfg.autocast):
                # pseudo_source_ann = ema(source_imgs).softmax(1)
                target_ann, _ = ema(target_imgs)
                target_ann = target_ann.softmax(1)

                mix_source_domain = (random.random() >= (iter/cfg.max_iters))

                if mix_source_domain:
                    sample_data = next(source_train_dataloader)
                    sample_imgs = sample_data["imgs"] if "imgs" in sample_data else [sample_data["img"]]
                    sample_imgs = [im.to(device) for im in sample_imgs]

                    sample_ann, _ = ema(sample_imgs)
                    sample_ann = sample_ann.softmax(1)
                    hard_ann = sample_data["ann"]
                    # if random.random() < (iter / (cfg.max_iters / 2)):
                    #     sample_ann, _ = ema(sample_imgs)
                    #     sample_ann = sample_ann.softmax(1)
                    # else:
                    #     sample_ann = sample_data["ann"]
                    #     sample_ann = sample_ann.to(device)
                    #     sample_ann = torch.nn.functional.one_hot(sample_ann, 25)
                    #     sample_ann = sample_ann.permute(0, 3, 1, 2).float()
                    # ann_prob, hard_ann = sample_ann.max(1)
                    # hard_ann[ann_prob < 0.968] = 0
                else:
                    sample_data = next(target_train_dataloader)
                    sample_imgs = sample_data["imgs"] if "imgs" in sample_data else [sample_data["img"]]
                    sample_imgs = [im.to(device) for im in sample_imgs]
                    sample_ann, _ = ema(sample_imgs)
                    sample_ann = sample_ann.softmax(1)
                    ann_prob, hard_ann = sample_ann.max(1)
                    hard_ann[ann_prob < 0.968] = 0
                
                for idx in range(0, cfg.train_batch_size):
                    mix_cat = rcm.get_mix_cat_id(cateList=torch.unique(hard_ann).tolist())
                    
                    if mix_cat != 0:
                        mix_idx = next((i for i in range(0, cfg.train_batch_size) if mix_cat in hard_ann[i]), None)
                        mix_mask = hard_ann[mix_idx]
                        mix_mask[mix_mask != mix_cat] = 0
                        mix_mask4imgs = mix_mask.unsqueeze(0).repeat(3, 1, 1)
                        mix_mask4soft_ann = mix_mask.unsqueeze(0).repeat(len(categories), 1, 1)    
                        mix_imgs = [im[mix_idx] for im in sample_imgs] # if mix_source_domain else [im[mix_idx] for im in source_imgs]
                        mix_ann = sample_ann[mix_idx] # if mix_source_domain else target_ann[mix_idx]

                        for i in range(0, len(target_imgs)):
                            target_imgs[i][idx][mix_mask4imgs != 0] = mix_imgs[i][mix_mask4imgs != 0]
                            if cfg.num_masks > 0:
                                erased_mask = (erased_target_imgs[i][idx] != 0).all(dim=0)
                                erased_mask = erased_mask.unsqueeze(0).repeat(3, 1, 1)
                                erased_target_imgs[i][idx][erased_mask] = target_imgs[i][idx][erased_mask]
                        target_ann[idx][mix_mask4soft_ann != 0] = mix_ann[mix_mask4soft_ann != 0]

                    view_ann = target_ann[idx].argmax(dim=0)
                    segmap = view_ann.cpu().squeeze(0).numpy()
                    color_map = palette[segmap]
                    show_im = Image.fromarray(color_map)
                    show_im.save(f'{tb_dir}/pseudo_label_{idx}.png')

                # max_pseudo_source_ann = torch.max(pseudo_source_ann, 1)
                # max_pseudo_target_ann = torch.max(target_ann, 1)
                # for cat in categories:
                #     source_confidences = max_pseudo_source_ann.values[max_pseudo_source_ann.indices == cat.id].flatten()
                #     source_confidences.sort()  # min to max
                #     if len(source_confidences) > 0:
                #         source_class_weight[cat.id] = -source_confidences[int(len(source_confidences) * 0.8)].log()

                #     target_confidences = max_pseudo_target_ann.values[max_pseudo_target_ann.indices == cat.id].flatten()
                #     target_confidences.sort()  # min to max
                #     if len(target_confidences) > 0:
                #         target_class_weight[cat.id] = -target_confidences[int(len(target_confidences) * 0.8)].log()
            
            with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                upsampled_logits, _ = model.forward(images=target_imgs)
                target_loss = target_criterion.compute(upsampled_logits, target_ann)
            scaler.scale(target_loss).backward()

            if cfg.num_masks > 0:
                with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                    upsampled_logits, _ = model(erased_target_imgs)
                    erased_target_loss = target_criterion.compute(upsampled_logits, target_ann)
                scaler.scale(erased_target_loss).backward()

            target_batch_num += 1
            target_batch_loss += target_loss.item()

            # source_class_weight.fill_(0)
            # target_class_weight.fill_(0)
                
        if iter % cfg.train_interval == 0:
            source_train_loss = source_batch_loss / cfg.train_interval
            source_train_miou = source_batch_miou / cfg.train_interval
            target_train_loss = target_batch_loss / target_batch_num
            source_batch_loss, source_batch_miou, target_batch_loss, target_batch_num = 0.0, 0.0, 0.0, 0.0
            
            writer.add_scalar('Source Loss/Train', source_train_loss, iter)
            writer.add_scalar('Source mIoU/Train', source_train_miou, iter)
            writer.add_scalar('Target Loss/Train', target_train_loss, iter)
            print(f"[Iter {iter}] Training Source Loss: {source_train_loss}, Source Mean_iou: {source_train_miou}, Target Loss: {target_train_loss}")
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if iter % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device_name):
                source_val_loss, source_val_miou, _, source_iou_list = source_validator.validate()
                target_val_loss, target_val_miou, _, target_iou_list = target_validator.validate()

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'source_best_miou': source_val_miou,
                    'target_best_miou': target_val_miou,
                    'iteration': iter
                }
                torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/latest_model_checkpoint_{currentTime}.pth")
                if source_val_miou + target_val_miou > source_best_miou + target_best_miou:
                    for file in os.listdir(f'logs/{exp_name}_{currentTime}'):
                        if file.startswith(f"best_model_checkpoint_iter"):
                            os.remove(f"logs/{exp_name}_{currentTime}/{file}")
                    torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/best_model_checkpoint_iter{iter}_{currentTime}.pth")
                    source_best_miou = source_val_miou
                    target_best_miou = target_val_miou

                writer.add_scalar('Source Loss/Validation', source_val_loss, iter)
                writer.add_scalar('Source mIoU/Validation', source_val_miou, iter)
                writer.add_scalar('Target Loss/Validation', target_val_loss, iter)
                writer.add_scalar('Target mIoU/Validation', target_val_miou, iter)
                print(f"Validation Source Loss: {source_val_loss}, Source Mean_iou: {source_val_miou}")
                print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': source_iou_list}))
                print(f"Validation Target Loss: {target_val_loss}, Target Mean_iou: {target_val_miou}")
                print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': target_iou_list}))
            gc.collect()
    writer.close()

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2 or len(sys.argv) == 3
    cfg = TrainingConfig_UDA.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    checkpoint = sys.argv[2] if len(sys.argv) == 3 else None
    log_dir = None if checkpoint is None else checkpoint.split('/')[-2]
    main(cfg, exp_name, checkpoint, log_dir)