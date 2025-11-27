import torch
from utils import convert_mask_to_class
from losses import combined_loss, dice_per_class, dice_per_class_3D, hausdorff_distance, ASSD
import numpy as np

def train_2D(model,train_loader,validation_loader,name,num_epochs=10, lr=1e-4, max_no_upgrade=10):
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

    return train_loss, validation_losses

def test_2D(model,test_loader,name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = combined_loss
    model.load_state_dict(torch.load(name+".pth", map_location=device))
    model.eval()

    test_loss = 0
    dice_per_class_total = np.zeros((5,))

    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X, test_y = test_X.to(device), convert_mask_to_class(test_y).to(device)
            outputs = model(test_X)
            test_loss += criterion(outputs, test_y).item()
            dice_per_class_total += dice_per_class(outputs, test_y).cpu().numpy()
    test_loss /= len(test_loader)
    print(f"Loss on test dataset: {test_loss:.4f}")
    print(f"Dice per class on test dataset: {dice_per_class_total / len(test_loader)}")
    return test_loss, dice_per_class_total / len(test_loader)

def test_3D(model, X_test_3D, y_test_3D, name):
    
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load(name+".pth", map_location=device))
    model.eval()
  
    dice = torch.zeros(5).to(device)
    hd = torch.zeros(5).to(device)
    assd = torch.zeros(5).to(device)
    
    with torch.no_grad():
        for i in range(len(X_test_3D)):
            target = convert_mask_to_class(y_test_3D[i].to(device))
            pred = model.forward_volume(X_test_3D[i].to(device))
            dice += dice_per_class_3D(pred, target)
            hd += hausdorff_distance(pred, target)
            assd += ASSD(pred, target)
    
    return dice/len(X_test_3D), hd/len(X_test_3D), assd/len(X_test_3D)




