import torch
from torch.utils.data import TensorDataset, DataLoader
from models import U_Net, AttU_Net
from utils import prepare_data, graph, convert_mask_to_class
from losses import combined_loss
import numpy as np



def train_2D(model, name, train_size, validation_size, test_size, num_epochs=10, lr=1e-4, max_no_upgrade=10):
    """
    model : Model we want to train
    name : Name of the weights we will save
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

    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = combined_loss

    num_batches = len(train_loader)

    best_val_loss = np.inf

    nb_no_upgrade = 0  # We store the number of epochs since when the accuracy on the validation set did not improve
    # if we don't improve for 3 epochs, we stop the process

    train_loss = []

    validation_losses = []

    for epoch in range(num_epochs):
        train_losses_on_epoch = 0
        if nb_no_upgrade < max_no_upgrade:
            for i, (batch_X, batch_y) in enumerate(train_loader, 1):

                batch_X = batch_X.to(device)
                batch_y = convert_mask_to_class(batch_y).to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                train_losses_on_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(train_losses_on_epoch / num_batches)

            # Validation après chaque epoch
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for val_X, val_y in validation_loader:
                    val_X, val_y = val_X.to(device), convert_mask_to_class(val_y).to(
                        device
                    )
                    outputs = model(val_X)
                    val_loss += criterion(outputs, val_y).item()

            val_loss /= len(validation_loader)
            validation_losses.append(val_loss)
            print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1} training loss: {train_losses_on_epoch / num_batches:.4f}")
            model.train()

            print(
                f"Previous best loss on validation set : {best_val_loss:.4f} | Loss on validation set at this epoch : {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), name+".pth")

                nb_no_upgrade = 0

            else:
                nb_no_upgrade += 1

    # Test on the test set
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X, test_y = test_X.to(device), convert_mask_to_class(test_y).to(device)
            outputs = model(test_X)
            test_loss += criterion(outputs, test_y).item()

    test_loss /= len(test_loader)
    print(f"Loss on test dataset: {test_loss:.4f}")

    return train_loss, validation_losses,test_loss

train_loss,validation_losses, test_loss = train_2D(
    model=AttU_Net(1, 5),
    name ="attU_Net_weights10",
    train_size=14,
    validation_size=2,
    test_size=4,
    num_epochs=10,
    lr=1e-4,
    max_no_upgrade=100
)

graph(train_loss,validation_losses)
