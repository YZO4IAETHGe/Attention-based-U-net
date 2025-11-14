import nibabel as nib
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import os


def normalize(img, h, w):
    if img.shape[0] > h:
        start_h = (img.shape[0] - h) // 2
        start_w = (img.shape[1] - w) // 2
        return img[None, start_h : start_h + h, start_w : start_w + w]
    else:
        start_h = (h - img.shape[0]) // 2
        start_w = (w - img.shape[1]) // 2
        pad = np.zeros((h, w))
        pad[start_h : start_h + img.shape[0], start_w : start_w + img.shape[1]] = img
        return pad[None, :, :]


def prepare_data(train_size, validation_size, data_path, rand = False):
    """
    To avoid bias, we aim to split the data at the patient level, ensuring that slices from the same patient
    do not appear in both the training and validation sets.
    We also normalize the data to have the same size and transform the list into numpy array.

    train_size : Number of images we want to take in our train set
    validation_size : Number of images we want to take in our validation set
    test_size : Number of images we want to take in our test set
    """

    data_X = []
    data_y = []

    # Keeping only the images for which we have the mask
    for i in range(40):

        try:
            path = os.path.join(data_path, "MR-dataset", f"{i+1:02d}-T1DUAL-mask.nii.gz")
            data_y.append(
                nib.load(path).get_fdata()
            )
            path = os.path.join(data_path, "MR-dataset", f"{i+1:02d}-T1DUALin-src.nii.gz")
            data_X.append(
                nib.load(path).get_fdata()
            )
            #data_y.append(
            #    nib.load(f"/content/drive/MyDrive/Cours/data/MR-dataset/{i+1:02d}-T1DUAL-mask.nii.gz").get_fdata()
            #)
            #data_X.append(
            #    nib.load(f"/content/drive/MyDrive/Cours/data/MR-dataset/{i+1:02d}-T1DUALin-src.nii.gz").get_fdata()
            #)
        except:
            pass

    # Get all the indices of the data
    indices = list(range(len(data_X)))
    if not(rand):
        random.seed(42)
    # Shuffle the indices
    random.shuffle(indices)
    print(indices)
    # Get random indices for the separation of the data
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + validation_size]
    test_idx = indices[train_size + validation_size :]

    X_train = []
    y_train = []
    X_validation = []
    y_validation = []
    X_test = []
    y_test = []

    # Split of the data
    for i in train_idx:
        for j in range(data_X[i].shape[2]):
            X_train.append(data_X[i][:, :, j])
            y_train.append(data_y[i][:, :, j])

    for i in val_idx:
        for j in range(data_X[i].shape[2]):
            X_validation.append(data_X[i][:, :, j])
            y_validation.append(data_y[i][:, :, j])

    for i in test_idx:
        for j in range(data_X[i].shape[2]):
            X_test.append(data_X[i][:, :, j])
            y_test.append(data_y[i][:, :, j])

    # Normalization of the data
    for i in range(len(X_train)):
        X_train[i] = normalize(X_train[i], 256, 256)
        y_train[i] = normalize(y_train[i], 256, 256)

    for i in range(len(X_validation)):
        X_validation[i] = normalize(X_validation[i], 256, 256)
        y_validation[i] = normalize(y_validation[i], 256, 256)

    for i in range(len(X_test)):
        X_test[i] = normalize(X_test[i], 256, 256)
        y_test[i] = normalize(y_test[i], 256, 256)

    # Convert data to array
    X_train = np.array(X_train, dtype=np.float32)
    X_validation = np.array(X_validation, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_validation = np.array(y_validation, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print("X_train.shape : ", X_train.shape)
    print("X_validation.shape : ", X_validation.shape)
    print("X_test.shape : ", X_test.shape)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_validation_tensor = torch.tensor(X_validation)
    y_validation_tensor = torch.tensor(y_validation)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    return (
        X_train_tensor,
        X_validation_tensor,
        X_test_tensor,
        y_train_tensor,
        y_validation_tensor,
        y_test_tensor,
    )

def convert_mask_to_class(mask):
    """
    mask: Tensor (B,1,H,W) avec valeurs dans [0, 80, 160, 240, 255]
    retourne: Tensor (B,1,H,W) avec valeurs entières dans [0,4]
    """
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask_squeezed = mask.squeeze(1)  # devient (B,H,W)
    else:
        mask_squeezed = mask

    mapping = torch.tensor([0.0, 80.0, 160.0, 240.0, 255.0], device=mask.device)
    result = torch.zeros_like(mask_squeezed, dtype=torch.long)
    for i, val in enumerate(mapping):
        result[mask_squeezed == val] = i

    return result.unsqueeze(1)  # remet le canal : (B,1,H,W)


def predict_mask(model, x):
    logits = model(x)
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    values = torch.tensor([0.0, 80.0, 160.0, 240.0, 255.0], device=preds.device)
    gray_mask = values[preds]  # (B,H,W)
    return gray_mask



def accuracy(output, target):
    if output.shape != target.shape:
        raise ValueError(
            f"Dimensions incompatibles : output.shape={output.shape}, target.shape={target.shape}"
        )

    # Exemple de calcul d'accuracy simple
    correct = (output == target).sum().item()
    total = target.numel()
    return correct / total

def graph(train_loss,validation_losses):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Premier subplot : Train Loss
    ax1.plot(range(len(train_loss)), train_loss, marker="o", color="red", label="Train Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss per Epoch")
    ax1.legend()
    ax1.grid(True)

    # Deuxième subplot : Validation Loss
    ax2.plot(
        range(len(validation_losses)),
        validation_losses,
        marker="o",
        color="blue",
        label="Validation Loss",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss per Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()  # ajuste l'espacement pour éviter que les titres/labels se chevauchent
    plt.show()

def compare_predictions_ground_truth(model, weights_name, input_path, output_path):

    model.load_state_dict(torch.load(weights_name, map_location = torch.device('cpu')))

    # Charger le fichier .nii.gz
    img = nib.load(input_path)
    input= img.get_fdata()

    img = nib.load(output_path)
    output = img.get_fdata()

    couche = 10
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
    output_hat = predict_mask(model,input1)
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