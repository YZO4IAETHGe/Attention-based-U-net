import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss


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

    target = target.squeeze(1)

    loss = F.cross_entropy(pred, target)

    # Dice loss (avec sigmoid interne)
    # dice = dice_loss(pred, target)
    dice = DiceLoss(mode="multiclass")
    dice_loss_value = dice(pred, target)

    loss = loss * bce_weight + dice_loss_value * (1 - bce_weight)
    return loss
