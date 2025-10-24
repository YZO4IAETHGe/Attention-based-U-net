import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    # Sigmoid et clamp pour éviter les NaN
    pred = torch.sigmoid(pred)
    pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )

    return 1 - dice


def calc_loss(pred, target, bce_weight=0.5):
    # BCE avec logits (avant sigmoid)
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Dice loss (avec sigmoid interne)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss
