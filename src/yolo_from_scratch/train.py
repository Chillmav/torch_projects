import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    intersection_over_union, non_max_suppression, mean_average_precision,
    cellboxes_to_boxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint)


from loss import YoloLoss

seed = 123
torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True, check_nan=False)
#Hyperparams

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/home/chillmaw/Documents/pochodnia/torch_projects/src/yolo_from_scratch/data/images"
LABEL_DIR = "/home/chillmaw/Documents/pochodnia/torch_projects/src/yolo_from_scratch/data/labels"

transform = v2.Compose([v2.Resize((448, 448)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

def train_fn(train_loader, model, optimizer, loss_fn):

    loop = tqdm(train_loader, leave=True)
    sum_loss = []

    for batch_idx, (x, y) in enumerate(loop):

        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        sum_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress_bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(sum_loss) / len(sum_loss)}")

def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("/home/chillmaw/Documents/pochodnia/torch_projects/src/yolo_from_scratch/data/8examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset("/home/chillmaw/Documents/pochodnia/torch_projects/src/yolo_from_scratch/data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()