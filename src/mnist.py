from torchvision.datasets import FashionMNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import matplotlib.pyplot as plt

train_data = FashionMNIST("C:/python/torch_projects/data", download=True, train=True, 
                          transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_data = FashionMNIST("C:/python/torch_projects/data", download=True, train=False, 
                         transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

class MLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x: torch.Tensor):
        
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
device = "cpu"
mlp = MLP().to(device)

# Training:

LR = 1e-3
BATCH_SIZE = 64

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=LR)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

def train_loop(train_loader, mlp, loss_fn, optimizer):
    
    size = len(train_loader.dataset)
    mlp.train()
    
    for batch, (X, y) in enumerate(train_loader):
        # Compute prediction and loss
        pred = mlp(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=LR)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, mlp, loss_fn, optimizer)
    test_loop(test_loader, mlp, loss_fn)
print("Done!")