import os
import cv2
import numpy as np
import warnings
import json
import random
from PIL import Image
from typing import List, Optional
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from engine.transform import Composition, Transform
from engine.category import Category

warnings.filterwarnings("ignore", category=UserWarning)

class RareCategoryManager:
    def __init__(
        self,
        categories: List[Category],
        rcs_path: str,
        temperature: float,
    ) -> None:
        with open(rcs_path) as f:
            data = json.load(f)

        self.stems = {cat.id: [] for cat in categories}
        self.category_probs = torch.zeros(len(categories))
        # ignore = [True if cat.id in rcs_cfg.ignore_ids else False for cat in categories]
        for d in data:
            # filename = Path(d["filename"]).stem
            count = np.array(d["count"])
            for cat in categories:
                if count[cat.id]:
                    self.stems[cat.id].append(d["filename"])
            self.category_probs += count
        self.consumable_stems = deepcopy(self.stems)
        # self.category_probs[ignore] = 0
        self.category_probs /= self.category_probs.sum()

        self.apply_temperature(temperature)

        self.length = len(data)

    def apply_temperature(self, temperature: float) -> None:
        self.sampling_probs = ((1 - self.category_probs) / temperature).exp()
        self.sampling_probs /= self.sampling_probs[self.category_probs != 0].sum()
        self.sampling_probs[self.category_probs == 0] = 0

    def get_rare_cat_id(self) -> int:
        return np.random.choice(
            [i for i in range(len(self.sampling_probs))],
            replace=True,
            p=self.sampling_probs.numpy(),
        )

    def get_stems(self, i: int) -> List[Path]:
        if len(self.consumable_stems[i]) == 0:
            # print(f"Already trained all images in category {i}!")
            self.consumable_stems[i] = deepcopy(self.stems[i])
        return self.consumable_stems[i]

class RLMD(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, rcm: Optional[RareCategoryManager], transforms: List[Transform]):
        self.img_paths = list()
        self.ann_paths = list()
        if rcm == None:
            for file in os.listdir(img_dir):
                self.img_paths.append(f'{img_dir}/{file}')
                self.ann_paths.append(f'{ann_dir}/{file[:-4]}.png')
                if ann_dir != None and os.path.exists(f'{ann_dir}/{file[:-4]}.png') == False:
                    self.img_paths.pop()
                    self.ann_paths.pop()
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.rcm = rcm
        self.transforms = Composition(transforms)

    def __len__(self):
        if self.rcm == None:
            return len(self.img_paths)
        else:
            return self.rcm.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.rcm == None:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
        else:
            random_cat_id = self.rcm.get_rare_cat_id()
            stems = self.rcm.get_stems(random_cat_id)
            stem = random.choice(stems)
            stems.remove(stem)
            img_path = f'{self.img_dir}/{stem.split("/")[-1][:-4]}.jpg'
            ann_path = stem

        return self.transforms.transform(
            {
                "img_path": img_path,
                "ann_path": ann_path
            }
        )
    
class CityscapesHDR(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, rcm: Optional[RareCategoryManager], transforms: List[Transform]):
        super().__init__()
        self.img_paths = list()
        self.ann_paths = list()
        if rcm == None:
            for folder in os.listdir(img_dir):
                for file in os.listdir(f'{img_dir}/{folder}'):
                    if os.path.exists(f'{img_dir}/{folder}/{file}') == True and os.path.exists(f'{ann_dir}/{folder}/{file[:-17]}_gtFine_labelTrainIds.png') == True:
                        self.img_paths.append(f'{img_dir}/{folder}/{file}')
                        self.ann_paths.append(f'{ann_dir}/{folder}/{file[:-17]}_gtFine_labelTrainIds.png')
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.rcm = rcm
        self.transforms = Composition(transforms)

    def __len__(self):
        if self.rcm == None:
            return len(self.img_paths)
        else:
            return self.rcm.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.rcm == None:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
        else:
            random_cat_id = self.rcm.get_rare_cat_id()
            stems = self.rcm.get_stems(random_cat_id)
            stem = random.choice(stems)
            stems.remove(stem)
            img_path = f'{self.img_dir}/{stem.split("/")[-1].split("_")[0]}/{stem.split("/")[-1].replace("gtFine_labelTrainIds", "leftImg16bit")}'
            ann_path = stem
        
        return self.transforms.transform(
            {
                "img_path": img_path,
                "ann_path": ann_path
            }
        )

class CityscapesLDR(Dataset):
    def __init__(self, img_dir, ann_dir, rcm: Optional[RareCategoryManager], transforms: List[Transform]):
        super().__init__()
        self.img_paths = list()
        self.ann_paths = list()
        if rcm == None:
            for folder in os.listdir(img_dir):
                for file in os.listdir(f'{img_dir}/{folder}'):
                    prefix = '_'.join(file.split("_")[:-1])
                    if os.path.exists(f'{img_dir}/{folder}/{file}') == True and os.path.exists(f'{ann_dir}/{folder}/{prefix}_gtFine_labelTrainIds.png') == True:
                        self.img_paths.append(f'{img_dir}/{folder}/{file}')
                        self.ann_paths.append(f'{ann_dir}/{folder}/{prefix}_gtFine_labelTrainIds.png')
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.rcm = rcm
        self.transforms = Composition(transforms)

    def __len__(self):
        if self.rcm == None:
            return len(self.img_paths)
        else:
            return self.rcm.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.rcm == None:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
        else:
            random_cat_id = self.rcm.get_rare_cat_id()
            stems = self.rcm.get_stems(random_cat_id)
            stem = random.choice(stems)
            stems.remove(stem)
            img_path = f'{self.img_dir}/{stem.split("/")[-1].split("_")[0]}/{stem.split("/")[-1].replace("gtFine_labelTrainIds", "leftImg8bit")}'
            ann_path = stem
        
        return self.transforms.transform(
            {
                "img_path": img_path,
                "ann_path": ann_path
            }
        )
    
def get_dataset(
    dataset_name: str,
    img_dir: str,
    ann_dir: str,
    rcm: Optional[RareCategoryManager],
    transforms: List[Transform]
):
    dataset = globals().get(dataset_name)
    assert dataset, f"There is no {dataset} dataset in datakiader.py!"
    return dataset(img_dir, ann_dir, rcm, transforms)

class InfiniteDataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool,
    ) -> None:
        self.dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self.iterator

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)