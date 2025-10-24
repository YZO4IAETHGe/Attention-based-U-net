import nibabel
import matplotlib.pyplot as plt


def print_in_mask_out(in_path, mask_path, out_path):

    img_in = nibabel.load(in_path)
    mask = nibabel.load(mask_path)
    img_out = nibabel.load(out_path)

    data_in = img_in.get_fdata()
    data_mask = mask.get_fdata()
    data_out = img_out.get_fdata()

    print("Shape de l'image :", data_in.shape)

    # Choisir une coupe au milieu du volume
    slice_index = data_in.shape[1] // 2  # plan axial au centre

    # Créer une figure avec 3 sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Afficher chaque image
    axes[0].imshow(data_in[:, slice_index, :], cmap="gray")
    axes[0].set_title("Image d'entrée")
    axes[0].axis("off")

    axes[1].imshow(data_mask[:, slice_index, :], cmap="gray")
    axes[1].set_title("Masque")
    axes[1].axis("off")

    axes[2].imshow(data_out[:, slice_index, :], cmap="gray")
    axes[2].set_title("Image de sortie")
    axes[2].axis("off")

    plt.suptitle(f"Coupe axiale {slice_index}")
    plt.tight_layout()
    plt.show()


in_path = "data/MR-dataset/01-T1DUALin-src.nii.gz"
mask_path = "data/MR-dataset/01-T1DUAL-mask.nii.gz"
out_path = "data/MR-dataset/01-T1DUALout-src.nii.gz"

print_in_mask_out(in_path, mask_path, out_path)
