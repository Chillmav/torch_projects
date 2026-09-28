import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

generator = torch.Generator().manual_seed(42)

IMAGES_PATH = "C:/python/torch_projects/src/object_detection/head_localization/data/images/"
ANNOTATIONS = "C:/python/torch_projects/src/object_detection/head_localization/data/annotations.csv"

class AnimalDataset(Dataset):
    
    def __init__(self, images_path, annotations):
        
        self.df = pd.read_csv(annotations)
        self.images_path = images_path
        self.annotations = annotations
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
    def __len__(self):
        
        return len(self.df) 
    
    def __getitem__(self, index):
        
        row = self.df.iloc[index] 
        filename = row["filename"]
        xmin = row["xmin"]
        ymin = row["ymin"]
        xmax = row["xmax"]
        ymax = row["ymax"]
        im = Image.open(self.images_path+filename).convert("RGB").resize((224,224))
        x = self.transform(im)
        y =  torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        
        return (x, y)
    
dataset = AnimalDataset(IMAGES_PATH, ANNOTATIONS)
train_length = int(round((len(dataset)*0.8)))
test_length = len(dataset) - train_length
lengths = [train_length, test_length]
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths, generator=generator)

