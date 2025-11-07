from U_Net import U_Net
from torchsummary import summary
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import numpy as np
from losses import predict_mask


def print_in_mask_out(in_path, out_path, mask_path=None):

    img_in = nib.load(in_path)
    img_out = nib.load(out_path)

    data_in = img_in.get_fdata()
    data_out = img_out.get_fdata()

    if mask_path:
        mask = nib.load(mask_path)
        data_mask = mask.get_fdata()

    print("Shape de l'image :", data_in.shape)

    # Choisir une coupe au milieu du volume
    slice_index = data_in.shape[1] // 2  # plan axial au centre

    # Créer une figure avec 3 sous-graphiques côte à côte
    if mask_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(data_in[:, slice_index, :], cmap="gray")
        axes[0].set_title("Image d'entrée")
        axes[0].axis("off")

        axes[1].imshow(data_mask[:, slice_index, :], cmap="gray")
        axes[1].set_title("Masque")
        axes[1].axis("off")

        axes[2].imshow(data_out[:, slice_index, :], cmap="gray")
        axes[2].set_title("Image de sortie")
        axes[2].axis("off")

    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].imshow(data_in[:, slice_index, :], cmap="gray")
        axes[0].set_title("Image d'entrée")
        axes[0].axis("off")

        axes[1].imshow(data_out[:, slice_index, :], cmap="gray")
        axes[1].set_title("Image de sortie")
        axes[1].axis("off")

    # Afficher chaque image

    plt.suptitle(f"Coupe axiale {slice_index}")
    plt.tight_layout()
    plt.show()


# in_path = "data/MR-dataset/01-T1DUALin-src.nii.gz"
# mask_path = "data/MR-dataset/01-T1DUAL-mask.nii.gz"
# out_path = "data/MR-dataset/01-T1DUALout-src.nii.gz"

# print_in_mask_out(in_path, mask_path, out_path)


def show_result():

    Unet = U_Net(1, 5)

    Unet.load_state_dict(
        torch.load("model_weights.pth", map_location=torch.device("cpu"))
    )

    # Charger le fichier .nii.gz
    img = nib.load("data/MR-dataset/39-T1DUALin-src.nii.gz")
    input = img.get_fdata()

    img = nib.load("data/MR-dataset/39-T1DUAL-mask.nii.gz")
    output = img.get_fdata()

    couche = 10

    valeurs_uniques = np.unique(output)

    print(valeurs_uniques)

    input1 = torch.tensor(input, dtype=torch.float32)
    input1 = input1.permute(2, 0, 1)
    input1 = input1[couche, :, :]
    output = torch.tensor(output, dtype=torch.float32)
    output = output.permute(2, 0, 1)
    output = output[couche, :, :]
    input1 = (input1.unsqueeze(0)).unsqueeze(0)
    print(input1.shape)
    output_hat = predict_mask(Unet, input1)
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

    for ax in axes:
        ax.axis("off")
    plt.show()
