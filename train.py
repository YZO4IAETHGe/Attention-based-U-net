import nibabel as nib
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from U_Net import U_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = []
y = []
for i in range(40):
    data = (nib.load(f"data/MR-dataset/{i+1:02d}-T1DUALin-src.nii.gz")).get_fdata()
    data2 = (nib.load(f"data/MR-dataset/{i+1:02d}-T1DUALout-src.nii.gz")).get_fdata()
    for j in range(data.shape[2]):
        X.append(data[:,:,j])
        y.append(data2[:,:,j])


def normalize(img, h,w):
    if img.shape[0]>h:
        start_h = (img.shape[0] - h)//2
        start_w = (img.shape[1] - w)//2
        return img[start_h:start_h+h,start_w:start_w+w]
    else :
        start_h = (h-img.shape[0])//2
        start_w = (w-img.shape[1])//2
        pad = np.zeros((h,w))
        pad[start_h:start_h+img.shape[0],start_w:start_w+img.shape[1]] = img
        return pad

for i in range(len(X)):
    X[i] = normalize(X[i],288,288)
    y[i] = normalize(y[i],288,288)

X = np.array(X, dtype=np.float32)   
y = np.array(y, dtype=np.float32)   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=10, shuffle=True)



model = U_Net(1,1)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

num_batches = len(train_loader)
num_epochs = 10

for epoch in range(num_epochs):
    for i, (batch_X, batch_y) in enumerate(train_loader, 1):

        batch_X = batch_X[:, None, :, :] 

        batch_X = batch_X.to(device) 
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        outputs = outputs.squeeze()
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Calcul du pourcentage
        percent_done = (i / num_batches) * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Batch {i}/{num_batches}, Loss: {loss.item():.4f}, Progress: {percent_done:.1f}%")


    torch.save(model.state_dict(), "model_weights.pth")


