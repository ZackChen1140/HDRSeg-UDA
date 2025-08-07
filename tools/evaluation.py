import sys
import os
import gc
import pandas as pd
import math
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# import evaluate

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from engine.segformer import get_model
from engine.dataloader import get_dataset, RLMD, InfiniteDataloader
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from engine.metric import Metrics
from engine.validator import Validator
from configs.config import EvaluationConfig


def main(cfg: EvaluationConfig, checkpoint: str):
    # dataset_info = pd.read_csv(cfg.category_csv)
    # categories = dataset_info['name'].to_list()
    categories = Category.load(cfg.category_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    assert len(cfg.image_roots) == len(cfg.label_roots), f"Inconsistent number of dataset paths."

    dataloaders = [
        DataLoader(
            dataset=get_dataset(
                dataset_name=cfg.dataset, 
                img_dir=cfg.image_roots[idx],
                ann_dir=cfg.label_roots[idx],
                rcm=None,
                transforms=val_transforms
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            pin_memory=cfg.pin_memory
        )
        for idx in range(0, len(cfg.image_roots))
    ]

    # metric = evaluate.load("mean_iou", keep_in_memory=True)
    metric = Metrics(num_categories=len(categories), ignore_ids=255, nan_to_num=0)

    segconfig = SegformerConfig().from_pretrained("nvidia/mit-b0")
    segconfig.num_labels = len(categories)
    
    model = get_model(cfg.model, segconfig)

    validators = [
        Validator(
            dataloader=dataloaders[idx], 
            model=model, 
            device=device, 
            metric=metric, 
            crop_size=cfg.crop_size,
            stride=cfg.stride,
            num_classes=len(categories),
            mode='slide',
            ignore_index=cfg.ignore_index[idx]
        )
        for idx in range(0, len(dataloaders))
    ]

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    model.eval()
    with torch.no_grad():
        for idx in range(0, len(validators)):
            loss, miou, _, iou_list = validators[idx].validate()
            print(f'Validation on Domain {idx}')
            print(f"Loss: {loss}, Mean_iou: {miou}")
            print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': iou_list}))
            print(' & '.join([f'{iou_list[idx] * 100:.2f}' for idx in range(0, len(categories))]))

        # source_images_root = '/home/rvl/hdd/rvl/Datasets/bdd100k/rainy/val/images_hdr'
        # source_labels_root = '/home/rvl/hdd/rvl/Datasets/bdd100k/rainy/val/labels'
        # target_images_root = '/home/rvl/hdd/rvl/Datasets/bdd100k/rainy/val/images_hdr_old'
        # target_labels_root = '/home/rvl/hdd/rvl/Datasets/bdd100k/rainy/val/labels_old'

        # file_list = [x[:17] for x in os.listdir('/home/rvl/hdd/rvl/Datasets/bdd100k_seg_maps/color_labels/val')]

        # n = 0
        # name_list, miou_list = validators[2].frame_wise_validate()
        # for idx in range(0, len(name_list)):
            # if miou_list[idx] < 0.35:
                # n += 1
                # os.remove(f'/home/rvl/hdd/rvl/Datasets/rlmd_hdr/night/val/labels/{name_list[idx]}.png')
            # print(f"filename: {name_list[idx]}, Mean IoU: {miou_list[idx]}")

        # print(n)

        # threshold = 0.4
        # for idx in range(0, len(name_list)):
        #     print(f"filename: {name_list[idx]}, Mean IoU: {miou_list[idx]}")
        #     if miou_list[idx] < threshold and name_list[idx] not in file_list:
        #         print(name_list[idx])
        #         shutil.copy(
        #             f'{source_images_root}/{name_list[idx]}.png',
        #             f'{target_images_root}/{name_list[idx]}.png'
        #         )
        #         os.remove(f'{source_images_root}/{name_list[idx]}.png')
        #         shutil.copy(
        #             f'{source_labels_root}/{name_list[idx]}.png',
        #             f'{target_labels_root}/{name_list[idx]}.png'
        #         )
        #         os.remove(f'{source_labels_root}/{name_list[idx]}.png')

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3
    cfg = EvaluationConfig.load(sys.argv[1])
    checkpoint = sys.argv[2]
    main(cfg, checkpoint)