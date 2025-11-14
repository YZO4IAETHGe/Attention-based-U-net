from models import U_Net, AttU_Net
from torchsummary import summary
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from losses import predict_mask
Unet = U_Net(1,5)
AttUnet =AttU_Net(1,5)
Unet.load_state_dict(torch.load("model_weights10.pth", map_location=torch.device('cpu')))
AttUnet.load_state_dict(torch.load("attU_Net_weights100.pth", map_location=torch.device('cpu')))



# Charger le fichier .nii.gz
img = nib.load("data/MR-dataset/34-T1DUALin-src.nii.gz")
input= img.get_fdata()

img = nib.load("data/MR-dataset/34-T1DUAL-mask.nii.gz")
output = img.get_fdata()

couche = 10
import numpy as np
valeurs_uniques = np.unique(output)

print(valeurs_uniques)

input1 = torch.tensor(input, dtype=torch.float32)
input1 = input1.permute(2, 0, 1)
input1 = input1[couche,:,:]
output = torch.tensor(output, dtype=torch.float32)
output = output.permute(2, 0, 1)
output = output[couche,:,:]
input1 = (input1.unsqueeze(0)).unsqueeze(0)
print(input1.shape)
output_hat = predict_mask(Unet,input1)
output_hat = output_hat.squeeze()
print(output_hat.shape)
output_hat = output_hat.detach().numpy()
input1 = input1.squeeze()

print(input1.shape)
print(output_hat.shape)
print(output.shape)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))


axes[0].imshow(input1[:, :], cmap="gray")
axes[0].set_title("in")

axes[1].imshow(output_hat[:, :], cmap="gray")
axes[1].set_title("Unet(input)")

axes[2].imshow(output[:, :], cmap="gray")
axes[2].set_title("out")

for ax in axes: ax.axis("off")
plt.show()