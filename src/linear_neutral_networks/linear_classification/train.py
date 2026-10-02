import torch
from loss import CrossLoss
from model import SoftmaxClassificationModel
from torch.utils.data import DataLoader
from dataset import train_data, test_data
from torch.optim import SGD

loss_fn = CrossLoss()

LR = 1e-3
BATCH_SIZE = 1
EPOCH = 100


model = SoftmaxClassificationModel()

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)
optim = SGD(model.parameters(), LR)
loss_fn = CrossLoss()

def train_loop(data, loss_fn, model, optim):

    model.train()
    total_loss = []
    for batch_idx, (X, y) in enumerate(data):

        pred = model(X)
        loss = loss_fn(y, pred)
        total_loss.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    print(f"AVG loss: {sum(total_loss) / len(total_loss)}")

def test_loop(data, model):

        model.eval
        data_size = len(data.dataset)
        acc = 0
        for batch_idx, (X, y) in enumerate(data):

            with torch.no_grad():
                pred = model(X)
                pred_label = torch.argmax(pred, dim=1)
                y_label = torch.argmax(y, dim=1)
                if pred_label == y_label:
                     acc += 1

        print(f"Total accuracy on validation set: {acc/data_size}")


for e in range(EPOCH):

    print(f"Epoch: {e+1}_____________________________________")

    train_loop(train_loader, loss_fn, model, optim)
    test_loop(test_loader, model)
                

