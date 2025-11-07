import nibabel as nib
import numpy as np
import torch
import random


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


def prepare_data(train_size, validation_size, test_size):
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
            data_y.append(
                nib.load(f"data/MR-dataset/{i+1:02d}-T1DUAL-mask.nii.gz").get_fdata()
            )
            data_X.append(
                nib.load(f"data/MR-dataset/{i+1:02d}-T1DUALin-src.nii.gz").get_fdata()
            )
        except:
            pass

    # Get all the indices of the data
    indices = list(range(len(data_X)))

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
        X_train[i] = normalize(X_train[i], 288, 288)
        y_train[i] = normalize(y_train[i], 288, 288)

    for i in range(len(X_validation)):
        X_validation[i] = normalize(X_validation[i], 288, 288)
        y_validation[i] = normalize(y_validation[i], 288, 288)

    for i in range(len(X_test)):
        X_test[i] = normalize(X_test[i], 288, 288)
        y_test[i] = normalize(y_test[i], 288, 288)

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
