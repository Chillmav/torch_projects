import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset import train_dataset, validation_dataset
from matplotlib.patches import Patch
from PIL import ImageDraw
from torchvision.ops import generalized_box_iou_loss
from torchvision.transforms.functional import to_pil_image

BATCH_SIZE = 2
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    # (N, 4) where N is the number of bboxes
        
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1] # slicing preserves the shape of a tensor (N, 1) instead of (N)
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    else:
        raise NotImplementedError("Only corners format is supported.")    
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return (intersection / (box1_area + box2_area - intersection + 1e-6)).mean()

class Detector(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.conv = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten()
        )
        
        self.linear = nn.Sequential(
            
            nn.Linear(16*7*7, 300),
            nn.ReLU(),
            nn.Linear(300, 50),
            nn.ReLU(),
            nn.Linear(50, 4),
            nn.Sigmoid()
        )
        
    def forward(self, x): # [BATCH_SIZE, 3, 224, 224]
        
        x = self.conv(x)
        box_params = self.linear(x)
        
        # Decode a lower corner and positive extents into valid (x1, y1, x2, y2).
        # The minimum extent keeps corners distinct even when sigmoid saturates.
        min_extent = 1e-4
        lower = box_params[..., :2] * (1 - min_extent)
        upper = lower + min_extent + (1 - lower - min_extent) * box_params[..., 2:]
        bbox = torch.cat((lower, upper), dim=-1)
        
        return bbox
    
model = Detector()

# def detect_loss(loss_fn, pred, y, iou_bonus=True):
#     if iou_bonus:
#         return loss_fn(pred, y) + (1 -  intersection_over_union(pred, y, box_format="corners"))
#     else:
#         return loss_fn(pred, y)

loss_fn = generalized_box_iou_loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train_loop(train_loader, model, loss_fn, optimizer):
    
    size = len(train_loader.dataset)
    model.train()
    iou_sum = 0.0
    window_samples = 0
    for batch, (X, y) in enumerate(train_loader):
        pred = model(X)
        loss = loss_fn(pred, y, reduction="mean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        iou_sum += intersection_over_union(pred, y).item() * len(X)
        window_samples += len(X)
        if batch % 200 == 0:
            avg_iou = iou_sum / window_samples
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], avg iou from last 50 batches: {avg_iou}")
            avg_iou = 0
            
def test_loop(dataloader, model, loss_fn):

    model.eval()
    num_batches = len(dataloader)
    test_loss =  0
    iou = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred, y, reduction="mean").item()
            iou += intersection_over_union(pred, y)
            if batch == num_batches - 1:
                image = to_pil_image(X[0])
                draw = ImageDraw.Draw(image)
                def pixel_box(box):
                    return [round(float(v) * 224) for v in box]
                draw.rectangle(pixel_box(y[0]), outline="green", width=3)
                draw.rectangle(pixel_box(pred[0]), outline="red", width=3)
                plt.imshow(image)
                
    test_loss /= num_batches
    iou /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f}, Avg IoU: {iou} \n")
    plt.axis("off")
    plt.legend(handles=[
        Patch(edgecolor="red", facecolor="none", label="Predicted"),
        Patch(edgecolor="green", facecolor="none", label="Target"),
    ], loc="upper right")
    plt.show()
    
EPOCHS = 10

for e in range(EPOCHS):
    print(f"Epoch {e+1} ________________________________")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
    
