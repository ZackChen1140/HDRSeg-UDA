from typing import Tuple, Optional, List, Union
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
    ignore_index: List[int]

    model: str
    train_batch_size: int
    val_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    pretrain_path: Optional[str]

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
    autocast: bool

@dataclass
class TrainingConfig_UDA(Serializable):
    dataset: str
    category_csv: str

    rcs_path: Optional[str]
    source_train_images_root: str
    source_train_labels_root: str
    target_train_images_root: List[str]
    source_val_images_root: str
    source_val_labels_root: str
    target_val_images_root: List[str]
    target_val_labels_root: List[str]
    ignore_index: List[List[int]]

    model: str
    train_batch_size: int
    val_batch_size: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    pretrain_path: Optional[str]
    
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
    mix_num: Optional[int]
    num_masks: int

    seed: int
    num_workers: int
    pin_memory: bool
    autocast: bool

@dataclass
class EvaluationConfig(Serializable):
    dataset: str
    category_csv: str

    image_roots: List[str]
    label_roots: List[str]
    ignore_index: List[List[int]]
    model: str
    batch_size: int

    max_intensity: Optional[float]
    contrast_stretch: Optional[List[str]]
    img_proc_params: Optional[list]
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]]
    pin_memory: bool