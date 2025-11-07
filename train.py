import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from U_Net import U_Net
from utils import prepare_data, accuracy
from time import time
from utils import prepare_data
from losses import convert_mask_to_class, combined_loss


def train_2D(train_size, validation_size, test_size, num_epochs, lr):
    """
    train_size : Number of images we want to take in our train set
    validation_size : Number of images we want to take in our validation set
    test_size : Number of images we want to take in our test set
    num_epochs : Number of epochs on which we want to train our model
    lr : Learning rate used by the optimizer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        X_train_tensor,
        X_validation_tensor,
        X_test_tensor,
        y_train_tensor,
        y_validation_tensor,
        y_test_tensor,
    ) = prepare_data(train_size, validation_size, test_size)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

    model = U_Net(1, 5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = combined_loss

    num_batches = len(train_loader)

    best_accuracy = 0

    nb_no_upgrade = 0  # We store the number of epochs since when the accuracy on the validation set did not improve
    # if we don't improve for 3 epochs, we stop the process

    train_loss = []
    train_losses_on_epoch = 0

    validation_accuracies = []

    for epoch in range(num_epochs):
        if nb_no_upgrade < 3:
            for i, (batch_X, batch_y) in enumerate(train_loader, 1):

                tm = time()

                batch_X = batch_X.to(device)
                batch_y = convert_mask_to_class(batch_y).to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                train_losses_on_epoch += loss.item
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(train_losses_on_epoch / num_batches)

            # Validation après chaque epoch
            model.eval()
            val_accuracy = 0

            with torch.no_grad():
                for val_X, val_y in validation_loader:
                    val_X, val_y = val_X.to(device), convert_mask_to_class(val_y).to(
                        device
                    )
                    outputs = model(val_X)
                    val_accuracy += accuracy(outputs, val_y)

            val_accuracy /= len(validation_loader)
            validation_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1} validation accuracy: {val_accuracy:.4f}")
            model.train()

            print(
                f"Previous best accuracy on validation set : {best_accuracy:.4f} | Accuracy on validation set at this epoch : {val_accuracy:.4f}"
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), "model_weights.pth")

                nb_no_upgrade = 0

            else:
                nb_no_upgrade += 1

    # Test on the test set
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model.eval()
    test_accuracy = 0

    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X, test_y = test_X.to(device), test_y.to(device)
            outputs = model(test_X)
            test_accuracy += accuracy(outputs, test_y)

    test_accuracy /= len(test_loader)
    print(f"Accuracy on test dataset: {test_accuracy:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Premier subplot : Train Loss
    ax1.plot(range(num_epochs), train_loss, marker="o", color="red", label="Train Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss per Epoch")
    ax1.legend()
    ax1.grid(True)

    # Deuxième subplot : Validation Accuracy
    ax2.plot(
        range(num_epochs),
        validation_accuracies,
        marker="o",
        color="blue",
        label="Validation Accuracy",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()  # ajuste l'espacement pour éviter que les titres/labels se chevauchent
    plt.show()


train_2D(14, 2, 4, 1, lr=1e-4)
