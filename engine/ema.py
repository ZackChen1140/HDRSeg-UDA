from typing import Dict
import torch

class PixelThreshold:
    def __init__(self, threshold: float = 0.968) -> None:
        assert isinstance(threshold, float)
        self.threshold = threshold

    def compute(
        self,
        logits: torch.Tensor,
        soft_ann: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prob, ann = soft_ann.max(1)
        # loss = criterion(pred, ann)
        if logits.shape[1] > 1:
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