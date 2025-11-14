from models import U_Net, AttU_Net
from train import train_2D, test_2D
from utils import compare_predictions_ground_truth, prepare_data, graph
from torch.utils.data import TensorDataset, DataLoader

(X_train_tensor,
X_validation_tensor,
X_test_tensor,
y_train_tensor,
y_validation_tensor,
y_test_tensor) = prepare_data(train_size = 14, validation_size=2, rand = False, data_path="data")

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


train_loss,validation_losses = train_2D(
    model=AttU_Net(1, 5),
    train_loader=train_loader,
    validation_loader=validation_loader,
    name ="attU_Net_10epochs",
    num_epochs=10,
    lr=1e-4,
    max_no_upgrade=100
)

test_loss,dice_classes = test_2D(
    model=AttU_Net(1, 5),
    test_loader=test_loader,
    name ="attU_Net_weightstest"
)

graph(train_loss,validation_losses)

compare_predictions_ground_truth(U_Net(1,5), "model_weights10.pth", "data/MR-dataset/34-T1DUALin-src.nii.gz", "data/MR-dataset/34-T1DUAL-mask.nii.gz")