import sys
import os
import gc
import pandas as pd
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import evaluate

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from engine.segformer import MultiPathSegformer, DualPathSegformer
from engine.dataloader import CityscapesLDR, CityscapesHDR_NLCS2, RLMD, RLMD_NLCS2, CityscapesHDR_GC3, InfiniteDataloader
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from engine.validator import Validator
from configs.config import TrainingConfig


def main(cfg: TrainingConfig, exp_name: str, checkpoint: str, log_dir: str):
    # dataset_info = pd.read_csv(cfg.category_csv)
    # categories = dataset_info['name'].to_list()
    categories = Category.load(cfg.category_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    val_transforms = [
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]

    val_dataset = RLMD_NLCS2(img_dir=cfg.val_images_root, ann_dir=cfg.val_labels_root, rcm=None, transforms=val_transforms, parameters=cfg.img_proc_params)
    # val_dataset = CityscapesLDR(img_dir=cfg.val_images_root, ann_dir=cfg.val_labels_root, rcm=None, transforms=val_transforms)

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
    segconfig.num_labels = 25
    
    model = DualPathSegformer(segconfig)
    # model = SegformerForSemanticSegmentation(config=segconfig)

    validator = Validator(val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')

    # optimizer = optim.AdamW(
    #     [
    #         {'name': 'backbone', 'params': model.encoder.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
    #         {'name': 'head', 'params': model.decoder.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
    #     ]
    # )
    # optimizer = optim.AdamW(
    #     [
    #         {'name': 'backbone', 'params': model.segformer.parameters(), 'lr': cfg.backbone_lr, 'weight_decay': cfg.weight_decay},
    #         {'name': 'head', 'params': model.decode_head.parameters(), 'lr': cfg.head_lr, 'weight_decay': cfg.weight_decay},
    #     ]
    # )

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # last_best = ckpt['best_miou']
    # last_iteration = ckpt['iteration']

    # batch_loss, batch_miou, batch_acc = 0, 0, 0
    model.eval()
    with torch.no_grad():
        # val_batch = 0.0
        # iou_list = None
        # batch_list = None
        # for inputs, labels in val_dataloader:
        #     inputs, labels = [im.to(device) for im in inputs], labels.to(device)
        #     # inputs, labels = inputs.to(device), labels.to(device)

        #     predicted, loss = model.forward(
        #         inputs=inputs,
        #         labels=labels
        #     )
        #     # outputs = model(pixel_values=inputs, labels=labels)
        #     # loss, logits = outputs.loss, outputs.logits
        #     # upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        #     # predicted = upsampled_logits.argmax(dim=1)
        #     metrics = metric._compute(
        #         predictions=predicted.cpu(),
        #         references=labels.cpu(),
        #         num_labels=segconfig.num_labels,
        #         ignore_index=255,
        #         reduce_labels=False,
        #     )
        #     val_batch += 1
        #     batch_loss += loss.item()
        #     batch_miou += metrics['mean_iou']
        #     batch_acc += metrics['mean_accuracy']
        #     if iou_list == None:
        #         iou_list = metrics['per_category_iou'].tolist()
        #         batch_list = [0.0 if math.isnan(v) else 1.0 for v in iou_list]
        #     else:
        #         for idx, iou in enumerate(metrics['per_category_iou'].tolist()):
        #             if not math.isnan(iou):
        #                 iou_list[idx] = iou if math.isnan(iou_list[idx]) else iou_list[idx] + iou
        #                 batch_list[idx] += 1

        # val_loss = batch_loss / val_batch
        # val_miou = batch_miou / val_batch
        # val_acc = batch_acc / val_batch
        # iou_list = [v / b for v, b in zip(iou_list, batch_list)]
        val_loss, val_miou, val_acc, iou_list = validator.validate()
        print(f"Validation Loss: {val_loss}, Mean_iou: {val_miou}, Mean accuracy: {val_acc}")
        print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': iou_list}))

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3
    cfg = TrainingConfig.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    checkpoint = sys.argv[2] if len(sys.argv) == 3 else None
    log_dir = None if checkpoint is None else checkpoint.split('/')[-2]
    main(cfg, exp_name, checkpoint, log_dir)