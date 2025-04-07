import torch
from torch import nn, Tensor


def compute_masks(img: Tensor, gt: Tensor, mask: Tensor, gt_div_tensor: Tensor) -> tuple[Tensor]:
    expanded_mask = mask.unsqueeze(1)
    crop_mask = expanded_mask.expand(-1, gt.shape[1], -1, -1)
    masked_gt = torch.where(crop_mask == 0, gt, torch.zeros_like(gt))
    masked_pred = torch.where(crop_mask == 0, img, torch.zeros_like(img))
    return masked_gt * gt_div_tensor, masked_pred * gt_div_tensor


def collate_fn_pad_full(batch: Tensor):
    max_h = max(img.shape[1] for img, gt in batch)  # Find max height in batch
    max_w = max(img.shape[2] for img, gt in batch)  # Find max width in batch

    padded_imgs = []
    padded_gts = []
    for img, gt in batch:
        padded_img = nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        padded_gt = nn.functional.pad(gt, (0, max_w - gt.shape[2], 0, max_h - gt.shape[1]))
        padded_imgs.append(padded_img)
        padded_gts.append(padded_gt)

    return torch.stack(padded_imgs), torch.stack(padded_gts)
