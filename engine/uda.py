import random
from typing import List, Dict, Tuple, Union, Any
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from transformers.modeling_outputs import BaseModelOutput

from engine.dataloader import RareCategoryManager
from engine.category import Category

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float,int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim() > 2:
            N, C, H, W = input.shape
            input = input.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
            target = target.view(-1)  # [N*H*W]

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).squeeze()  # [N*H*W]
            input = input[valid_mask]    # shape: [num_valid_pixels, C]
            target = target[valid_mask]  # shape: [num_valid_pixels]
            if target.numel() == 0:
                return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        logpt = logpt.view(-1)
        pt = logpt.exp().detach()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        loss = loss.view(N, H, W)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class PixelThreshold:
    def __init__(self, threshold: float = 0.968, focal_loss: FocalLoss = None) -> None:
        assert isinstance(threshold, float)
        self.threshold = threshold
        self.focal_loss = focal_loss

    def compute(
        self,
        logits: torch.Tensor,
        soft_ann: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prob, ann = soft_ann.max(1)

        if self.focal_loss is not None:
            loss = self.focal_loss.forward(input=logits, target=ann)
        elif logits.shape[1] > 1:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
            loss = loss_fct(logits, ann)
        elif logits.shape[1] == 1:
            valid_mask = ((ann >= 0) & (ann != 255)).float()
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits.squeeze(1), ann.float())
            loss = (loss * valid_mask)
        ge = prob >= self.threshold

        if ge.any():
            return loss[ge].mean()
        return torch.zeros(1, requires_grad=True).cuda()
    
class ClassMix:
    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        rcm: RareCategoryManager,
        categories: List[Category],
        source_dataloader: DataLoader = None,
        target_dataloader: DataLoader = None,
        ema: EMA = None,
        mix_num: int = 1
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.rcm = rcm
        self.categories = categories
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.ema = ema
        self.mix_num = mix_num

    def mix(
        self,
        mix_domain: int,
        imgs: List[torch.Tensor],
        ann: torch.Tensor,
        dis_label: torch.Tensor = None,
        erased_imgs: List[torch.Tensor] = None,
    ):
        sample_data = next(self.source_dataloader if mix_domain == 0 else self.target_dataloader)
        sample_imgs = sample_data["imgs"] if "imgs" in sample_data else [sample_data["img"]]
        sample_imgs = [im.to(self.device) for im in sample_imgs]
        sample_ann, _ = self.ema(sample_imgs)
        sample_ann = sample_ann.softmax(1)
        sample_imgs, sample_ann = [im.to(torch.device('cpu')) for im in sample_imgs], sample_ann.to(torch.device('cpu'))

        if mix_domain == 0:
            hard_ann = sample_data["ann"]
        else:
            ann_prob, hard_ann = sample_ann.max(1)
            hard_ann[ann_prob < 0.968] = 0

        for idx in range(0, self.batch_size):
            mix_cat_list = self.rcm.get_mix_cat_id(cateList=torch.unique(hard_ann).tolist(), mix_num=self.mix_num)
            
            for mix_cat in mix_cat_list:
                if mix_cat != 0:
                    mix_idx = next((i for i in range(0, self.batch_size) if mix_cat in hard_ann[i]), None)
                    mix_domain = sample_data["domain"][mix_idx] # to distinguish between rainy and night
                    mix_mask = hard_ann[mix_idx].clone()
                    mix_mask[mix_mask != mix_cat] = 0
                    mix_mask4imgs = mix_mask.unsqueeze(0).repeat(3, 1, 1)
                    mix_mask4soft_ann = mix_mask.unsqueeze(0).repeat(len(self.categories), 1, 1)    
                    mix_imgs = [im[mix_idx] for im in sample_imgs] # if mix_source_domain else [im[mix_idx] for im in source_imgs]
                    mix_ann = sample_ann[mix_idx] # if mix_source_domain else target_ann[mix_idx]

                    for i in range(0, len(imgs)):
                        imgs[i][idx][mix_mask4imgs != 0] = mix_imgs[i][mix_mask4imgs != 0]
                    ann[idx][mix_mask4soft_ann != 0] = mix_ann[mix_mask4soft_ann != 0]
                    if dis_label is not None:
                        dis_label[idx][mix_mask4soft_ann != 0] = mix_domain
            if erased_imgs is not None:
                for i in range(0, len(imgs)):
                    erased_mask = (erased_imgs[i][idx] != 0).all(dim=0)
                    erased_mask = erased_mask.unsqueeze(0).repeat(3, 1, 1)
                    erased_imgs[i][idx][erased_mask] = imgs[i][idx][erased_mask]
    
def compute_domain_discrimination_loss(
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    imgs: List[torch.Tensor],
    ann: torch.Tensor,
    dis_label: torch.Tensor,
    # domain_label: int, 
    domain_class_weight: torch.Tensor,
    crop_size: Tuple[int, int], 
    device: torch.device
) -> torch.Tensor:
    latent = model.encoder.forward(
        *imgs,
        output_attentions=model.config.output_attentions,
        output_hidden_states=True,
        return_dict=True
    )
    if isinstance(latent, BaseModelOutput):
        latent = latent.hidden_states
    dis_pred = discriminator(latent)
    dis_pred = F.interpolate(dis_pred, crop_size, mode="bilinear")

    dis_loss = F.binary_cross_entropy_with_logits(
        dis_pred,
        dis_label,
        domain_class_weight[ann][:, None, :],
    ).mean()

    dis_pred.to(torch.device('cpu'))

    return dis_loss