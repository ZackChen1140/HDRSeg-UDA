from typing import Tuple, Optional
from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class TrainingConfig(Serializable):
    category_csv: str
    rcs_path: Optional[str]

    train_images_root: str
    train_labels_root: str
    val_images_root: str
    val_labels_root: str

    img_proc_params: list

    train_batch_size: int
    val_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float

    max_iters: int
    train_interval: int
    val_interval: int

    rcs_temperature: float
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]]
    random_resize_ratio: Tuple[float, float]

    seed: int
    num_workers: int
    pin_memory: bool

@dataclass
class TuningConfig(Serializable):
    num_samples: int
    gpus_per_trial: float
    grace_period: int

    category_csv: str
    rcs_path: str

    train_images_root: str
    train_labels_root: str
    val_images_root: str
    val_labels_root: str

    img_proc_params: list

    train_batch_size: list
    val_batch_size: int
    backbone_lr: Tuple[float, float]
    head_lr: Tuple[float, float]
    weight_decay: Tuple[float, float]

    max_iters: int
    val_interval: int

    rcs_temperature: float
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    random_resize_ratio: Tuple[float, float]

    seed: int
    num_workers: int
    pin_memory: bool