import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ce_loss = torch.nn.CrossEntropyLoss()

def dice_per_class(pred, target, epsilon=1e-6):
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
    return dice_per_class

def dice_loss(pred, target, epsilon=1e-6):
    dice_class = dice_per_class(pred, target, epsilon)
    return 1 - dice_class.mean()

def combined_loss(pred, target, alpha=0.5):
    # Squeeze target pour cross-entropy et Dice
    if target.dim() == 4 and target.shape[1] == 1:
        target_squeezed = target.squeeze(1)
    else:
        target_squeezed = target

    dice = dice_loss(pred, target_squeezed)
    ce = torch.nn.functional.cross_entropy(pred, target_squeezed)
    return alpha * dice + (1 - alpha) * ce

def dice_per_class_3D(pred, target, epsilon=1e-6):
    """
    pred: (D, C, H, W)
    target: entiers (D, H, W)
    """
    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)

    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(pred_soft * target_onehot, dim = (0, 2, 3))
    cardinality = torch.sum(pred_soft + target_onehot, dim = (0, 2, 3))
    dice_per_class = (2.0 * intersection + epsilon) / (cardinality + epsilon)

    return dice_per_class

def contour_volume(x, class_idx):
    """
    x: logits (D, H, W)
    class_idx: int
    """
    volume_class = x == class_idx
    eroded_volume = binary_erosion(volume_class, structure=np.ones((3,3,3)))
    contour_volume = volume_class ^ eroded_volume
    return contour_volume

def hausdorff_distance(pred, target):
    """
    pred: logits (D, C, H, W)
    target: entiers (D, H, W)
    class_idx: int
    """
    hd = torch.zeros(pred.shape[1])
    pred_classes = torch.argmax(pred, dim=1)
    for class_idx in range(pred.shape[1]):
        
        pred_contour = contour_volume(pred_classes.cpu().numpy(), class_idx)
        target_contour = contour_volume(target.cpu().numpy(), class_idx)

        pred_points = np.argwhere(pred_contour)
        target_points = np.argwhere(target_contour)

        if len(pred_points) == 0 or len(target_points) == 0:
            hd[class_idx] = np.inf

        else:
            tree_pred = cKDTree(pred_points)
            tree_target = cKDTree(target_points)

            distances_target_to_pred, _ = tree_pred.query(target_points)
            distances_pred_to_target, _ = tree_target.query(pred_points)

            hausdorff_dist = max(distances_pred_to_target.max(), distances_target_to_pred.max())
            hd[class_idx] = hausdorff_dist
    return hd

def ASSD(pred, target):
    """
    pred: logits (D, C, H, W)
    target: entiers (D, H, W)
    class_idx: int
    """
    assd = torch.zeros(pred.shape[1])
    pred_classes = torch.argmax(pred, dim=1)
    for class_idx in range(pred.shape[1]):
        
        pred_contour = contour_volume(pred_classes.cpu().numpy(), class_idx)
        target_contour = contour_volume(target.cpu().numpy(), class_idx)

        pred_points = np.argwhere(pred_contour)
        target_points = np.argwhere(target_contour)

        if len(pred_points) == 0 or len(target_points) == 0:
            assd[class_idx] = np.inf

        else:
            tree_pred = cKDTree(pred_points)
            tree_target = cKDTree(target_points)

            distances_target_to_pred, _ = tree_pred.query(target_points)
            distances_pred_to_target, _ = tree_target.query(pred_points)

            assd_dist = (distances_pred_to_target.sum() + distances_target_to_pred.sum())/ (len(pred_points) + len(target_points))
            assd[class_idx] = assd_dist
    return assd