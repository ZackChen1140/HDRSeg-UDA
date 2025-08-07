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

from torch_optimizer import Lookahead
from ema_pytorch import EMA
from transformers import SegformerConfig

from engine.segformer import get_model
from engine.dataloader import get_dataset, RareCategoryManager, InfiniteDataloader
from engine.category import Category
from engine.uda import PixelThreshold, ClassMix, compute_domain_discrimination_loss
from engine import transform
from engine.misc import set_seed
from engine.metric import Metrics
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

    # Create for rare category sampling
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
        transform.Check(),
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
        transform.Check(),
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
        transform.Check(),
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
    print(f'{cfg.dataset} Target Validation Dataset Size: {target_val_dataset.__len__()}')
    assert source_train_dataset.__len__() > 0
    assert target_train_dataset.__len__() > 0
    assert source_val_dataset.__len__() > 0
    assert target_val_dataset.__len__() > 0

    source_class_weight = torch.zeros((len(categories)))
    target_class_weight = torch.zeros((len(categories)))

    metric = Metrics(num_categories=len(categories), nan_to_num=0)

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = len(categories)
    
    model = get_model(cfg.model, segconfig)
    ema = EMA(model, beta=0.999, update_after_step=-1)
    discriminator = get_model(f"{cfg.model}Discriminator", segconfig)

    base_optimizer = optim.AdamW(
        [
            {'name': 'backbone', 'params': model.encoder.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
            {'name': 'head', 'params': model.decoder.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
            {"name": "dicriminator", "params": discriminator.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay}
        ]
    )
    optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)

    source_validator = Validator(source_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide', ignore_index=cfg.ignore_index[0])
    target_validator = Validator(target_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide', ignore_index=cfg.ignore_index[1])

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer.optimizer, 1e-4, 1, 1500)
    poly_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer.optimizer, cfg.max_iters - 1500, 1)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer.optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[1500]
    )

    target_criterion = PixelThreshold(threshold=0.968)
    classmix = ClassMix(
        device=device,
        batch_size=cfg.train_batch_size,
        rcm=rcm,
        categories=categories,
        source_dataloader=source_train_dataloader,
        target_dataloader=target_train_dataloader,
        ema=ema,
        mix_num=cfg.mix_num
    )

    scaler = torch.GradScaler(device=device_name)

    model.to(device)
    ema.to(device)
    ema.ema_model.to(device)
    discriminator.to(device)
    torch.compile(model)
    torch.compile(ema.ema_model)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        source_last_best = ckpt['source_best_miou']
        target_last_best = ckpt['target_best_miou']
        last_iteration = ckpt['iteration']
        optimizer.optimizer.param_groups[0]["lr"] = scheduler._last_lr[0]
        optimizer.optimizer.param_groups[1]["lr"] = scheduler._last_lr[1]
    elif cfg.pretrain_path != "":
        ckpt = torch.load(cfg.pretrain_path)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.ema_model.load_state_dict(ckpt['model_state_dict'])

    ema_milestone = (cfg.max_iters + len(cfg.ema_update_intervals)) // len(cfg.ema_update_intervals)
    source_best_miou = 0.0 if checkpoint is None else source_last_best
    target_best_miou = 0.0 if checkpoint is None else target_last_best
    begin_iter = 0 if checkpoint is None else last_iteration
    source_batch_loss, target_batch_loss, source_dis_batch_loss, target_dis_batch_loss ,target_batch_num = 0.0, 0.0, 0.0, 0.0, 0.0
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

        metric.compute_and_accum(source_pred.cpu(), source_ann.cpu())
        source_batch_loss += source_loss.item()

        if iter % ema.update_every == 0:
            target_data = next(target_train_dataloader)
            target_imgs = target_data["imgs"] if "imgs" in target_data else [target_data["img"]]

            erased_target_imgs = None
            if cfg.num_masks > 0:
                erased_target_imgs = target_data["erased imgs"] if "erased imgs" in target_data else [target_data["erased img"]]

            with torch.no_grad(), torch.autocast(device_type=device_name, enabled=cfg.autocast):
                # obtain pseudo labels for target images
                target_imgs = [im.to(device) for im in target_imgs]
                target_ann, _ = ema(target_imgs)
                target_ann = target_ann.softmax(1)
                target_imgs, target_ann = [im.to(torch.device('cpu')) for im in target_imgs], target_ann.to(torch.device('cpu'))

                # save pseudo labels
                for idx in range(0, cfg.train_batch_size):
                    view_ann = target_ann[idx].argmax(dim=0)
                    segmap = view_ann.cpu().squeeze(0).numpy()
                    color_map = palette[segmap]
                    show_im = Image.fromarray(color_map)
                    show_im.save(f'{tb_dir}/pseudo_label_{idx}.png')
                
                # obtain discrimination labels
                source_dis_label = torch.zeros(target_ann.shape)
                target_dis_label = torch.zeros(target_ann.shape)
                source_dis_label[:] = 0
                for idx in range(0, cfg.train_batch_size):
                    target_dis_label[idx, :] = target_data["domain"][idx]

                # class mix
                mix_domain = 0 if (random.random() > 0.5) else 1
                classmix.mix(
                    mix_domain=mix_domain,
                    imgs=target_imgs,
                    ann= target_ann,
                    dis_label=target_dis_label,
                    erased_imgs=erased_target_imgs
                )

                # obtain pseudo labels for source images to compute class weights
                pseudo_source_ann, _ = ema(source_imgs)
                pseudo_source_ann = pseudo_source_ann.softmax(1)
                pseudo_source_ann = pseudo_source_ann.to(torch.device('cpu'))

                max_pseudo_source_ann = torch.max(pseudo_source_ann, 1)
                max_pseudo_target_ann = torch.max(target_ann, 1)
                for cat in categories:
                    source_confidences = max_pseudo_source_ann.values[max_pseudo_source_ann.indices == cat.id].flatten()
                    source_confidences.sort()  # min to max
                    if len(source_confidences) > 0:
                        source_class_weight[cat.id] = -source_confidences[int(len(source_confidences) * 0.8)].log()

                    target_confidences = max_pseudo_target_ann.values[max_pseudo_target_ann.indices == cat.id].flatten()
                    target_confidences.sort()  # min to max
                    if len(target_confidences) > 0:
                        target_class_weight[cat.id] = -target_confidences[int(len(target_confidences) * 0.8)].log()
            
            with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                target_imgs, target_ann = [im.to(device) for im in target_imgs], target_ann.to(device)
                upsampled_logits, _ = model.forward(images=target_imgs)
                target_loss = target_criterion.compute(upsampled_logits, target_ann)
            scaler.scale(target_loss).backward()

            if cfg.num_masks > 0:
                with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                    erased_target_imgs = [im.to(device) for im in erased_target_imgs]
                    upsampled_logits, _ = model(erased_target_imgs)
                    erased_target_loss = target_criterion.compute(upsampled_logits, target_ann)
                    erased_target_imgs, upsampled_logits = [im.to(torch.device('cpu')) for im in erased_target_imgs], upsampled_logits.to(torch.device('cpu'))
                scaler.scale(erased_target_loss).backward()

            with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                source_dis_label, source_class_weight = source_dis_label.to(device), source_class_weight.to(device)
                source_dis_loss = compute_domain_discrimination_loss(model, discriminator, source_imgs, source_ann, source_dis_label, source_class_weight, cfg.crop_size, device)
                source_dis_label, source_class_weight = source_dis_label.to(torch.device('cpu')), source_class_weight.to(torch.device('cpu'))
            scaler.scale(source_dis_loss).backward()

            with torch.autocast(device_type=device_name, enabled=cfg.autocast):
                target_hard_ann = target_ann.argmax(1)
                target_hard_ann, target_dis_label, target_class_weight = target_hard_ann.to(device), target_dis_label.to(device), target_class_weight.to(device)
                target_dis_loss = compute_domain_discrimination_loss(model, discriminator, target_imgs, target_hard_ann, target_dis_label, target_class_weight, cfg.crop_size, device)
                target_hard_ann, target_dis_label, target_class_weight = target_hard_ann.to(torch.device('cpu')), target_dis_label.to(torch.device('cpu')), target_class_weight.to(torch.device('cpu'))
            scaler.scale(target_dis_loss).backward()
            
            source_class_weight.fill_(0)
            target_class_weight.fill_(0)

            target_batch_num += 1
            target_batch_loss += target_loss.item()
            source_dis_batch_loss += source_dis_loss.item()
            target_dis_batch_loss += target_dis_loss.item()
                
        if iter % cfg.train_interval == 0:
            source_train_loss = source_batch_loss / cfg.train_interval
            source_train_miou = metric.get_and_reset()["IoU"].mean()
            target_train_loss = target_batch_loss / target_batch_num
            source_train_dis_loss = source_dis_batch_loss / target_batch_num
            target_train_dis_loss = target_dis_batch_loss / target_batch_num
            source_batch_loss, target_batch_loss, source_dis_batch_loss, target_dis_batch_loss, target_batch_num = 0.0, 0.0, 0.0, 0.0, 0.0
            
            writer.add_scalar('Source Loss/Train', source_train_loss, iter)
            writer.add_scalar('Source mIoU/Train', source_train_miou, iter)
            writer.add_scalar('Target Loss/Train', target_train_loss, iter)
            writer.add_scalar('Source Loss/Discriminator', source_train_dis_loss, iter)
            writer.add_scalar('Target Loss/Discriminator', target_train_dis_loss, iter)
            writer.add_scalar('Learning Rate', optimizer.optimizer.param_groups[0]['lr'], iter)

            print(
                f"[Iter {iter}] Training Source Loss: {source_train_loss}, "
                f"Source Mean_iou: {source_train_miou}, "
                f"Target Loss: {target_train_loss}, "
                f"Learning Rate: {optimizer.optimizer.param_groups[0]['lr']}"
            )
        
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
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
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
                
                if iter % (cfg.max_iters / 4) == 0:
                    torch.save(checkpoint, f"logs/{exp_name}_{currentTime}/checkpoint_iter_{iter}_{currentTime}.pth")

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