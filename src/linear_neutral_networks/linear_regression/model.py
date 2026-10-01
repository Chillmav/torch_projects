import torch
import torch.nn as nn
from dataset import train_data, test_data
from torch.utils.data import DataLoader

EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 10
REGULARIZATION_PARAMETER = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(13, 1)
        
    def forward(self, X):
        out = self.lin(X)
        return out


def loss_fn(out, y, model, regularize=False):

    loss = nn.functional.mse_loss(out, y)
    if regularize:
        loss += REGULARIZATION_PARAMETER * model.lin.weight.pow(2).sum() / 2
    return loss

def train_loop(train_loader, model, optimizer, loss_fn, weight_decay=False):

    total_loss = 0.0
    num_examples = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        out = model(X)
        loss = loss_fn(out, y, model, regularize=True)  # mean loss for this batch

        batch_size = X.size(0)
        num_examples += batch_size
        total_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Mean loss after {num_examples} examples: {total_loss / num_examples}")


def mae_loss(out, y):

    return abs(out - y)

def main():

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data)
    model = LinearRegression().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), LR)
    for epoch in range(EPOCHS):

        print(f"EPOCH {epoch}: ------------------------------------------------------------")
        
        train_loop(train_loader, model, optimizer, loss_fn, weight_decay=True)

    total_loss = 0
    length = len(test_loader.dataset)
    model.eval()
    for batch_idx, (X, y) in enumerate(test_loader):
        with torch.no_grad():
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = mae_loss(out, y)
            total_loss += loss.item()

    print(f"MAE on validation set: {total_loss/length}")


if __name__ == "__main__":

    main()


