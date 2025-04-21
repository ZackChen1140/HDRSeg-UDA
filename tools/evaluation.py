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

from engine.segformer import get_model
from engine.dataloader import get_dataset, RLMD, InfiniteDataloader
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from engine.validator import Validator
from configs.config import TrainingConfig_UDA


def main(cfg: TrainingConfig_UDA, exp_name: str, checkpoint: str, log_dir: str):
    # dataset_info = pd.read_csv(cfg.category_csv)
    # categories = dataset_info['name'].to_list()
    categories = Category.load(cfg.category_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

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

    source_val_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.source_val_images_root, ann_dir=cfg.source_val_labels_root, rcm=None, transforms=val_transforms)
    target_val_dataset = get_dataset(dataset_name=cfg.dataset, img_dir=cfg.target_val_images_root, ann_dir=cfg.target_val_labels_root, rcm=None, transforms=val_transforms)

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

    metric = evaluate.load("mean_iou", keep_in_memory=True)

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = len(categories)
    
    model = get_model(cfg.model, segconfig)

    source_validator = Validator(source_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')
    target_validator = Validator(target_val_dataloader, model, device, metric, cfg.crop_size, cfg.stride, len(categories), mode='slide')

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    model.eval()
    with torch.no_grad():
        source_val_loss, source_val_miou, _, source_iou_list = source_validator.validate()
        target_val_loss, target_val_miou, _, target_iou_list = target_validator.validate()
        print(f"Validation Source Loss: {source_val_loss}, Source Mean_iou: {source_val_miou}")
        print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': source_iou_list}))
        print(f"Validation Target Loss: {target_val_loss}, Target Mean_iou: {target_val_miou}")
        print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': target_iou_list}))

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3
    cfg = TrainingConfig_UDA.load(sys.argv[1])
    exp_name = sys.argv[1].split('/')[-1][:-5]
    checkpoint = sys.argv[2] if len(sys.argv) == 3 else None
    log_dir = None if checkpoint is None else checkpoint.split('/')[-2]
    main(cfg, exp_name, checkpoint, log_dir)