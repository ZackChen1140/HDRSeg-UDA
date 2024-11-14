from typing import Any, Protocol, Optional, Dict, List, Tuple, Union
from torchvision.transforms import functional as F
from torchvision import transforms as T
#from engine.category import Category
import cv2
import torch
import numpy as np
import math
import random
from PIL import Image


class Transform(Protocol):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]: ...


class Composition:
    def __init__(self, transformations: List[Transform]) -> None:
        self.transformations = transformations

    def __repr__(self) -> str:
        print("Composition:")
        for i, transformation in enumerate(self.transformations):
            print(f"({i})", transformation)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transformation in self.transformations:
            data = transformation.transform(data)
        return data


class LoadImg:
    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = cv2.imread(data["img_path"])
        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data["img"] = F.to_tensor(image)
        return data


class NDArrayImgToTensor:
    def __init__(self, to_rgb: bool = True) -> None:
        self.to_rgb = to_rgb

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.to_rgb:
            data["img"] = cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB)
        data["img"] = F.to_tensor(data["img"])
        return data


class LoadAnn:
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ann = Image.open(data["ann_path"])
        print(ann.mode)
        data["ann"] = torch.from_numpy(np.asarray(ann).copy())[None, :].long()
        return data


class Identity:

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class Resize:
    def __init__(
        self, image_scale: Optional[Tuple[int, int]] = None, antialias: bool = True
    ) -> None:
        self.image_scale = image_scale
        self.antialias = antialias

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "img" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img"].shape[-2:]
            )

            data["img"] = F.resize(data["img"], _image_scale, antialias=self.antialias)

        if "img0" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img0"].shape[-2:]
            )

            data["img0"] = F.resize(data["img0"], _image_scale, antialias=self.antialias)
        
        if "img1" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img1"].shape[-2:]
            )

            data["img1"] = F.resize(data["img1"], _image_scale, antialias=self.antialias)

        if "img2" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["img2"].shape[-2:]
            )

            data["img2"] = F.resize(data["img2"], _image_scale, antialias=self.antialias)

        if "ann" in data:
            _image_scale = (
                self.image_scale if self.image_scale else data["ann"].shape[-2:]
            )
            data["ann"] = F.resize(
                data["ann"][:, None, :],
                _image_scale,
                interpolation=F.InterpolationMode.NEAREST,
            ).squeeze()

        return data


class RandomResizeCrop:
    def __init__(
        self,
        image_scale: Tuple[int, int],
        scale: Tuple[float, float],
        crop_size: Tuple[int, int],
        antialias: bool = True,
        # cat_ratio: float = 0.0,
        rare_cat_crop: bool = False,
        patient: int = 10,
        efficient: bool = False,
        efficient_interval: int = 10,
    ) -> None:
        self.image_scale = image_scale
        self.scale = scale
        self.crop_size = np.array(crop_size)
        self.antialias = antialias
        self.rare_cat_crop = rare_cat_crop
        self.patient = patient
        self.efficient = efficient
        if self.efficient:
            self.crop_sizes = np.linspace(self.crop_size, self.crop_size // 2, 6)
            self.efficient_interval = efficient_interval
            self.efficient_counter = 0

    def get_random_size(self):
        min_scale, max_scale = self.scale
        random_scale = random.random() * (max_scale - min_scale) + min_scale
        height = int(self.image_scale[0] * random_scale)
        width = int(self.image_scale[1] * random_scale)
        return height, width

    def get_crop_size(self):
        if self.efficient:
            crop_size = self.crop_sizes[self.efficient_counter // 8]
            self.efficient_counter += 1
            if self.efficient_counter == 48:
                self.efficient_counter = 0
        else:
            crop_size = self.crop_size
        return crop_size.astype(int)

    def get_random_crop(self, scaled_height, scaled_width, crop_size):
        crop_y0 = random.randint(0, scaled_height - crop_size[0])
        crop_x0 = random.randint(0, scaled_width - crop_size[1])

        return crop_y0, crop_x0

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        height, width = self.get_random_size()
        crop_size = self.get_crop_size()
        y0, x0 = self.get_random_crop(height, width, crop_size)
        if "ann" in data:
            if self.rare_cat_crop:
                assert (
                    "ann" in data
                ), "Category-ratio cropping is avaliable only when label is given!"
                if "random_cat_id" in data:
                    random_id = data["random_cat_id"]
                else:
                    random_id = random.choice(
                        data["ann"].unique(sorted=False)
                    )  # Choose a random category id in the label
                uncropped_ann = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()

                best_ratio = 0
                best_x0 = x0
                best_y0 = y0
                for _ in range(self.patient):
                    ann = uncropped_ann[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                    ratio = (
                        torch.where(ann == random_id)[0].shape[0]
                        / ann.flatten().shape[0]
                    )
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_x0 = x0
                        best_y0 = y0

                    y0, x0 = self.get_random_crop(height, width)

                x0 = best_x0
                y0 = best_y0
                data["ann"] = uncropped_ann[
                    y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]
                ]
            else:
                data["ann"] = F.resize(
                    data["ann"][:, None, :],
                    (height, width),
                    interpolation=F.InterpolationMode.NEAREST,
                ).squeeze()[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        if "img" in data:
            data["img"] = F.resize(
                data["img"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        if "img0" in data:
            data["img0"] = F.resize(
                data["img0"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
        
        if "img1" in data:
            data["img1"] = F.resize(
                data["img1"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
        
        if "img2" in data:
            data["img2"] = F.resize(
                data["img2"], (height, width), antialias=self.antialias
            )[:, y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]

        data["height"] = height
        data["width"] = width
        data["y0"] = y0
        data["x0"] = x0

        return data


class Normalize:
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = mean
        self.std = std

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "img" in data:
            data["img"] = F.normalize(data["img"], self.mean, self.std)
        if "img0" in data:
            data["img0"] = F.normalize(data["img0"], self.mean, self.std)
        if "img1" in data:
            data["img1"] = F.normalize(data["img1"], self.mean, self.std)
        if "img2" in data:
            data["img2"] = F.normalize(data["img2"], self.mean, self.std)

        return data


class ColorJitter:

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "img" in data:
            data["img"] = self.jitter(data["img"])
        if "img0" in data:
            data["img0"] = self.jitter(data["img0"])
        if "img1" in data:
            data["img1"] = self.jitter(data["img1"])
        if "img2" in data:
            data["img2"] = self.jitter(data["img2"])

        return data


class WeakAndStrong:
    def __init__(self, weak_transform: Transform, strong_transform: Transform) -> None:
        self.weak = weak_transform
        self.strong = strong_transform

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "img" in data:
            weak_img = self.weak.transform(data)["img"]
            data["strong_img"] = self.strong.transform(data)["img"]
            data["img"] = weak_img
        if "img0" in data:
            weak_img = self.weak.transform(data)["img0"]
            data["strong_img0"] = self.strong.transform(data)["img0"]
            data["img0"] = weak_img
        if "img1" in data:
            weak_img = self.weak.transform(data)["img1"]
            data["strong_img1"] = self.strong.transform(data)["img1"]
            data["img1"] = weak_img
        if "img2" in data:
            weak_img = self.weak.transform(data)["img2"]
            data["strong_img2"] = self.strong.transform(data)["img2"]
            data["img2"] = weak_img

        return data


class RandomGaussian:
    def __init__(self, p: float = 0.5, kernel_size: int = 3) -> None:
        self.p = p
        self.kernel_size = kernel_size

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() >= self.p and "img" in data:
            data["img"] = F.gaussian_blur(data["img"], self.kernel_size)
        if random.random() >= self.p and "img0" in data:
            data["img0"] = F.gaussian_blur(data["img0"], self.kernel_size)
        if random.random() >= self.p and "img1" in data:
            data["img1"] = F.gaussian_blur(data["img1"], self.kernel_size)
        if random.random() >= self.p and "img2" in data:
            data["img2"] = F.gaussian_blur(data["img2"], self.kernel_size)

        return data


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            if "img" in data:
                data["img"] = F.hflip(data["img"])
            if "img0" in data:
                data["img0"] = F.hflip(data["img0"])
            if "img1" in data:
                data["img1"] = F.hflip(data["img1"])
            if "img2" in data:
                data["img2"] = F.hflip(data["img2"])

            if "ann" in data:
                data["ann"] = F.hflip(data["ann"][:, None, :])[:, 0]

        return data


class RandomErase:
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: int = 0,
    ) -> None:
        self.erase = T.RandomErasing(p, scale, ratio, value)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "erased img" in data or "erased img0" in data:
            if "erased img" in data:
                data["erased img"] = self.erase(data["erased img"])
            if "erased img0" in data:
                data["erased img0"] = self.erase(data["erased img0"])
            if "erased img1" in data:
                data["erased img1"] = self.erase(data["erased img1"])
            if "erased img2" in data:
                data["erased img2"] = self.erase(data["erased img2"])
        else:
            if "img" in data:
                data["erased img"] = self.erase(data["img"])
            if "img0" in data:
                data["erased img0"] = self.erase(data["img0"])
            if "img1" in data:
                data["erased img1"] = self.erase(data["img1"])
            if "img2" in data:
                data["erased img2"] = self.erase(data["img2"])

        return data
