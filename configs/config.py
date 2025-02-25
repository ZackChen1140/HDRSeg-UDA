from typing import Tuple, Optional, List
from simple_parsing import Serializable
from dataclasses import dataclass

@dataclass
class TrainingConfig(Serializable):
    dataset: str
    category_csv: str

    rcs_path: Optional[str]
    train_images_root: str
    train_labels_root: str
    val_images_root: str
    val_labels_root: str

    model: str
    train_batch_size: int
    val_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float

    max_iters: int
    train_interval: int
    val_interval: int

    rcs_temperature: Optional[float]
    max_intensity: Optional[float]
    contrast_stretch: Optional[List[str]]
    img_proc_params: Optional[list]
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]]
    random_resize_ratio: Tuple[float, float]

    seed: int
    num_workers: int
    pin_memory: bool

@dataclass
class TrainingConfig_UDA(Serializable):
    dataset: str
    category_csv: str

    rcs_path: Optional[str]
    source_train_images_root: str
    source_train_labels_root: str
    target_train_images_root: str
    source_val_images_root: str
    source_val_labels_root: str
    target_val_images_root: str
    target_val_labels_root: str

    model: str
    train_batch_size: int
    val_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float

    max_iters: int
    train_interval: int
    val_interval: int
    ema_update_intervals: List[int]

    rcs_temperature: Optional[float]
    max_intensity: Optional[float]
    contrast_stretch: Optional[List[str]]
    img_proc_params: Optional[list]
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]]
    random_resize_ratio: Tuple[float, float]
    num_masks: int

    seed: int
    num_workers: int
    pin_memory: bool
    autocast: bool