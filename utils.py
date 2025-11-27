import nibabel as nib
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch.nn.functional as F


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

def prepare_data(train_size, validation_size, data_path, random_seed=None):
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
    indices = np.array(range(len(data_X)))
    if random_seed:
        random.seed(random_seed)
    # Shuffle the indices
    random.shuffle(indices)

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
    couche_test = []
    X_test_3D =[]
    y_test_3D = []

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
        couche_test.append(data_X[i].shape[2])
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

    couche = 0
    j = 0
    X_test_slices = []
    y_test_slices = []
    for i in range(len(X_test)):
        X_test[i] = normalize(X_test[i], 256, 256)
        y_test[i] = normalize(y_test[i], 256, 256)
        X_test_slices.append(X_test[i])
        y_test_slices.append(y_test[i])
        couche +=1
        if couche_test[j] == couche:
            X_test_3D.append(torch.tensor(np.stack(X_test_slices, axis=1).squeeze(0), dtype=torch.float32))
            y_test_3D.append(torch.tensor(np.stack(y_test_slices, axis=1).squeeze(0), dtype=torch.float32))
            X_test_slices = []
            y_test_slices = []
            couche = 0
            j+=1


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
        X_test_3D,
        y_test_3D
    )

def show_input_mask(input_path, output_path, slice_idx):

    # Charger les fichiers NIfTI
    img_in = nib.load(input_path)
    input_vol = img_in.get_fdata()

    img_out = nib.load(output_path)
    output_vol = img_out.get_fdata()

    # Extraire la même couche dans les deux volumes
    input_slice = input_vol[:, :, slice_idx]
    output_slice = output_vol[:, :, slice_idx]

    # Affichage côte à côte
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_slice, cmap='gray')
    plt.title("Input")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_slice, cmap='gray')
    plt.title("Output")
    plt.axis('off')

    plt.show()

def convert_mask_to_class(mask):
    """
    mask: Tensor de n'importe quelle dimension contenant les valeurs [0, 80, 160, 240, 255]
    retourne: Tensor de même shape contenant les classes [0,1,2,3,4]
    """

    mapping = torch.tensor([0.0, 80.0, 160.0, 240.0, 255.0], device=mask.device)

    result = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)

    for i, val in enumerate(mapping):
        result[mask == val] = i

    return result

def convert_logits_to_class(logits):

    if logits.dim() == 4:

        preds = torch.argmax(logits, dim=1)  # (B,H,W)
        values = torch.tensor([0.0, 80.0, 160.0, 240.0, 255.0], device=preds.device)
        gray_mask = values[preds]
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

    input_tensor = torch.tensor(input, dtype=torch.float32)
    input_tensor = input_tensor.permute(2, 0, 1)

    output_tensor = torch.tensor(output, dtype=torch.float32)
    output_tensor = output_tensor.permute(2, 0, 1)

    output_hat = model.forward_volume(input_tensor)

    print(type(output_hat))
    print(output_hat.shape)

    output_hat = convert_logits_to_class(output_hat)

    output_hat = output_hat.detach().numpy()

    display_volumes(input_tensor, output_tensor, output_hat)

def display_volumes(volume1, volume2, volume3):
    """
    Affiche 3 volumes côte à côte avec un slider unique pour explorer les slices.
    volume1, volume2, volume3 : np.array (D,H,W)
    """
    D = volume1.shape[0]
    assert volume2.shape[0] == D and volume3.shape[0] == D, "Tous les volumes doivent avoir le même nombre de slices"

    def view_slice(idx):
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(volume1[idx], cmap='gray', vmin=volume1.min(), vmax=volume1.max())
        axes[0].set_title("input")
        axes[0].axis('off')

        axes[1].imshow(volume2[idx], cmap='gray', vmin=volume2.min(), vmax=volume2.max())
        axes[1].set_title("target")
        axes[1].axis('off')

        axes[2].imshow(volume3[idx], cmap='gray', vmin=volume3.min(), vmax=volume3.max())
        axes[2].set_title("prediction")
        axes[2].axis('off')

        plt.show()
        display(slice_slider)

    slice_slider = widgets.IntSlider(min=0, max=D-1, value=0, description='Slice')
    slice_slider.observe(lambda change: view_slice(change['new']), names='value')

    # Affiche la première slice
    view_slice(0)

def predict_volume(model, weights_name, input_path):
    """
    Charge les poids, applique le modèle tranche par tranche et reconstruit un volume 3D.
    Retourne (nifti_img, volume_numpy) :
      - nifti_img : nib.Nifti1Image construit avec l'affine du fichier d'entrée
      - volume_numpy : ndarray shape (H, W, D) avec les valeurs des masques (0,80,160,240,255)
    """

    # charger les poids sur CPU (sécurisé si pas de GPU disponible)
    state = torch.load(weights_name, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()

    # charger le volume d'entrée
    img = nib.load(input_path)
    data = img.get_fdata()  # typiquement (H, W, D)

    if data.ndim < 3:
        raise ValueError(f"Le volume attendu doit avoir au moins 3 dimensions, got {data.ndim}")

    # choisir le device du modèle (si le modèle reste sur CPU ce sera 'cpu')
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    data_tensor = data_tensor.permute(2, 0, 1)

    logits = model.forward_volume(data_tensor)

    volume = convert_logits_to_class(logits).cpu().numpy()

    nifti_out = nib.Nifti1Image(volume, img.affine)

    return nifti_out, volume

def load_slice(input_path, slice_number):

    nii = nib.load(input_path)
    data = nii.get_fdata()  # (H,W,D)
    slice2d = data[:,:,slice_number]

    # Normalise et met au format (1,1,H,W)
    image = torch.tensor(slice2d, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

    return image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, image, class_idx):
        image = image.clone().detach().requires_grad_(True)

        outputs = self.model(image)
        probs = F.softmax(outputs, dim=1)
        score = probs[:, class_idx].sum()

        self.model.zero_grad()
        score.backward()

        # activation = (1, C, H', W')
        act = self.activations
        grad = self.gradients

        # pondération moyenne des gradients
        weights = grad.mean(dim=[2, 3], keepdim=True)

        # somme pondérée des feature maps
        cam = (weights * act).sum(dim=1).relu()

        cam = cam - cam.min()
        cam = cam / cam.max()

        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(image.shape[2], image.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        return cam.squeeze()

def overlay_cam(image, cam):
    img = image.squeeze().cpu().numpy()
    hm = cam.detach().cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.imshow(hm, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.show()

def overlay_cam_multi(image, cams, titles=None):
    """
    Affiche côte à côte plusieurs CAMs (ou saliency maps)
    en suivant le style de la fonction overlay_cam.

    image : tensor de l'image (1,H,W)
    cams  : liste de tensors de saliency maps
    titles : liste de titres optionnels
    """
    img = image.squeeze().cpu().numpy()

    n = len(cams)
    plt.figure(figsize=(20,4))

    for i in range(n):
        cam = cams[i].detach().cpu().numpy()

        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap="gray")
        plt.imshow(cam, cmap="jet", alpha=0.45)
        if titles is not None:
            plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def saliency_map(model, image, class_idx):
    image = image.clone().detach().requires_grad_(True)
    model.eval()
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)

    score = probs[:, class_idx].sum()
    score.backward()

    grad = image.grad.abs().squeeze()
    return grad