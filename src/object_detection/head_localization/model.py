import torch
from dataset import  train_dataset, validation_dataset
import torch.nn as nn

BATCH_SIZE = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Detector(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten()
        )
        
        self.linear = nn.Sequential(
            
            nn.Linear(16*14*14, 600),
            nn.ReLU(),
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Linear(100, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x): # [BATCH_SIZE, 3, 224, 224]
        
        x = self.conv(x)
        bbox = self.linear(x)
        
        return bbox
    
model = Detector()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train_loop(train_loader, model, loss_fn, optimizer):
    
    size = len(train_loader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), batch + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):

    model.eval()
    num_batches = len(dataloader)
    test_loss =  0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

EPOCHS = 10

for e in range(EPOCHS):
    print(f"Epoch {e} ________________________________")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
    
