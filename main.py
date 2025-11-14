from models import U_Net, AttU_Net
from torchsummary import summary
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import predict_mask, compare_predictions_ground_truth

compare_predictions_ground_truth(U_Net(1,5), "U_Net_weights30.pth", "data/MR-dataset/34-T1DUALin-src.nii.gz", "data/MR-dataset/34-T1DUAL-mask.nii.gz")