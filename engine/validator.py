import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from evaluate import EvaluationModule
from engine.metric import Metrics

class Validator:
    def __init__(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        # metric: EvaluationModule,
        metric: Metrics,
        crop_size: Tuple[int, int],
        stride: Tuple[int, int],
        num_classes: int,        
        mode: str
    ):
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.metric = metric
        self.crop_size = crop_size
        self.stride = stride
        self.num_classes = num_classes
        self.mode = mode
        assert mode == 'slide' or mode == 'basic', 'mode must be \'slide\' or \'basic\'.'

    def validate(self):
        avg_loss = 0
        avg_miou = 0
        avg_acc = 0
        iou_list = None
        batch_list = None
        for data in self.dataloader:
            imgs = data["imgs"] if "imgs" in data else [data["img"]]
            ann = data["ann"]
            imgs, ann = [im.to(self.device) for im in imgs], ann.to(self.device)
            logits = self.slide_inference(images=imgs)
            predicted = logits.argmax(dim=1)
            if self.num_classes > 1:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
                loss = loss_fct(logits, ann)
            elif self.num_classes == 1:
                valid_mask = ((ann >= 0) & (ann != 255)).float()
                loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits.squeeze(1), ann.float())
                loss = (loss * valid_mask).mean()
            
            # metrics = self.metric._compute(
            #     predictions=predicted.cpu(),
            #     references=ann.cpu(),
            #     num_labels=self.num_classes,
            #     ignore_index=255,
            #     nan_to_num=0,
            #     reduce_labels=False,
            # )
            self.metric.compute_and_accum(predicted, ann)

            avg_loss += loss.item()
            # avg_miou += metrics['mean_iou']
            # avg_acc += metrics['mean_accuracy']
            # if iou_list == None:
            #     iou_list = metrics['per_category_iou'].tolist()
            #     batch_list = [0.0 if math.isnan(v) else 1.0 for v in iou_list]
            # else:
            #     for idx, iou in enumerate(metrics['per_category_iou'].tolist()):
            #         if not math.isnan(iou):
            #             iou_list[idx] = iou if math.isnan(iou_list[idx]) else iou_list[idx] + iou
            #             batch_list[idx] += 1

        iou = self.metric.get_and_reset()["IoU"]
        
        avg_loss = avg_loss / len(self.dataloader)
        avg_miou = iou.mean()
        avg_acc = 0.0
        iou_list = iou
        # avg_miou = avg_miou / len(self.dataloader)
        # avg_acc = avg_acc / len(self.dataloader)
        # iou_list = [v / b for v, b in zip(iou_list, batch_list)]

        return avg_loss, avg_miou, avg_acc, iou_list

    def slide_inference(
        self,
        images: List[torch.Tensor],
    ):
        inputs = list()
        h_stride, w_stride = self.stride
        h_crop, w_crop = self.crop_size
        batch_size, _, h_img, w_img = images[0].size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = images[0].new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = images[0].new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                inputs = [image[:, :, y1:y2, x1:x2] for image in images]
                upsampled_logits, _ = self.model.forward(images=inputs)
                preds += F.pad(
                    upsampled_logits,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        return preds
    
    # def inference(self):