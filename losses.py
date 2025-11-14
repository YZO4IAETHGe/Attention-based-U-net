import torch
import torch.nn.functional as F
from torch import nn

ce_loss = torch.nn.CrossEntropyLoss()


def dice_loss(pred, target, epsilon=1e-6):
    """
    pred: logits (B, C, H, W)
    target: entiers (B, H, W)
    """
    # Squeeze si target a un canal
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)  # devient (B,H,W)

    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(pred_soft * target_onehot, dim=(0, 2, 3))
    cardinality = torch.sum(pred_soft + target_onehot, dim=(0, 2, 3))
    dice_per_class = (2.0 * intersection + epsilon) / (cardinality + epsilon)
    return 1 - dice_per_class.mean()


def combined_loss(pred, target, alpha=0.5):
    # Squeeze target pour cross-entropy et Dice
    if target.dim() == 4 and target.shape[1] == 1:
        target_squeezed = target.squeeze(1)
    else:
        target_squeezed = target

    dice = dice_loss(pred, target_squeezed)
    ce = torch.nn.functional.cross_entropy(pred, target_squeezed)
    return alpha * dice + (1 - alpha) * ce