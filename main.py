from U_Net import U_Net
from torchsummary import summary
import nibabel as nib
import matplotlib.pyplot as plt
import torch

Unet = U_Net(35,35)

#print(summary(Unet, (35, 256, 256)))


# Charger le fichier .nii.gz
img = nib.load("data/MR-dataset/01-T1DUALin-src.nii.gz")
input= img.get_fdata()

img = nib.load("data/MR-dataset/01-T1DUALout-src.nii.gz")
output = img.get_fdata()

input1 = torch.tensor(input, dtype=torch.float32)
input1 = input1.permute(2, 0, 1)
input1 = input1.unsqueeze(0)
output_hat = Unet(input1)
output_hat = output_hat.detach().numpy()

print(output_hat.shape)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))


axes[0].imshow(input[:, :, input.shape[2] // 2], cmap="gray")
axes[0].set_title("in")

axes[1].imshow(output_hat[0,:, :, output_hat.shape[2] // 2], cmap="gray")
axes[1].set_title("Unet(input)")

axes[2].imshow(output[:, :, output.shape[2] // 2], cmap="gray")
axes[2].set_title("out")

for ax in axes: ax.axis("off")
plt.show()