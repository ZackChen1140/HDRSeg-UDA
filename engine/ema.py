from typing import List, Dict, Tuple
import torch
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

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
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        # target = target.view(-1,1)
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
        # loss = criterion(pred, ann)

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
    # imgs = [im.to(device) for im in imgs]
    # ann = ann.to(device)
    # dis_label = dis_label.to(device)
    # domain_class_weight = domain_class_weight.to(device)

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
    # dis_label = torch.zeros(dis_pred.shape).cuda()
    # dis_label[:] = domain_label

    dis_loss = F.binary_cross_entropy_with_logits(
        dis_pred,
        dis_label,
        domain_class_weight[ann][:, None, :],
    ).mean()

    dis_pred.to(torch.device('cpu'))

    return dis_loss