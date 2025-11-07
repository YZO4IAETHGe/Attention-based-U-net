import nibabel as nib
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from U_Net import U_Net
from losses import combined_loss, convert_mask_to_class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = []
y = []
for i in range(40):
    data = (nib.load(f"data/MR-dataset/{i+1:02d}-T1DUALin-src.nii.gz")).get_fdata()
    try:
        data2 = (nib.load(f"data/MR-dataset/{i+1:02d}-T1DUAL-mask.nii.gz")).get_fdata()
        for j in range(data.shape[2]):
            X.append(data[:,:,j])
            y.append(data2[:,:,j])
    except:
        pass

def normalize(img, h,w):
    if img.shape[0]>h:
        start_h = (img.shape[0] - h)//2
        start_w = (img.shape[1] - w)//2
        return img[None,start_h:start_h+h,start_w:start_w+w]
    else :
        start_h = (h-img.shape[0])//2
        start_w = (w-img.shape[1])//2
        pad = np.zeros((h,w))
        pad[start_h:start_h+img.shape[0],start_w:start_w+img.shape[1]] = img
        return pad[None,:,:]

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
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)



model = U_Net(1,5)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
criterion = combined_loss

num_batches = len(train_loader)
num_epochs = 10
losses_list = []
for epoch in range(num_epochs):
    losses = 0
    for i, (batch_X, batch_y) in enumerate(train_loader, 1):

     
        batch_X = batch_X.to(device) 
        batch_y = convert_mask_to_class(batch_y).to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
    
    avg_loss = losses / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    losses_list.append(avg_loss)
        
    torch.save(model.state_dict(), "model_weights.pth")


plt.plot(range(1, num_epochs + 1), losses_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()
