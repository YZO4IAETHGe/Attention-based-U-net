import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from U_Net import U_Net
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = U_Net(1, 5)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = combined_loss

    num_batches = len(train_loader)
    losses_list = []
    for epoch in range(num_epochs):
        losses_epoch = 0
        for i, (batch_X, batch_y) in enumerate(train_loader, 1):

            batch_X = batch_X.to(device)
            batch_y = convert_mask_to_class(batch_y).to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_epoch += loss.item()

        avg_loss = losses_epoch / num_batches
        losses_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "model_weights.pth")
        
    plt.plot(range(1, num_epochs + 1), losses_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.show()

train_2D(14, 2, 4, 10, lr=1e-4)
