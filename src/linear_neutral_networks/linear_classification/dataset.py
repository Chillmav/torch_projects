from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.data import random_split

CSV_PATH = "/home/chillmaw/Documents/pochodnia/torch_projects/src/linear_neutral_networks/linear_classification/data/archive/Iris.csv"

class IrisDataset(Dataset):

    def __init__(self, csv_path, standardization=False):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.df =  self.df.dropna()

        if standardization:

            numeric_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

            means = self.df[numeric_columns].mean()
            stds = self.df[numeric_columns].std()

            self.df[numeric_columns] = (self.df[numeric_columns] - means) / stds

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        class_dict = {
            "Iris-setosa": torch.tensor([1, 0, 0]),
            "Iris-versicolor": torch.tensor([0, 1, 0]),
            "Iris-virginica": torch.tensor([0, 0, 1])
        }

        X = torch.tensor(
            [
                row["SepalLengthCm"], row["SepalWidthCm"], row["PetalLengthCm"], row["PetalWidthCm"]
            ], dtype=torch.float32
        )

        y = class_dict[row["Species"]]
        return X, y

dataset = IrisDataset(CSV_PATH, standardization=False)

length = len(dataset)
train_length = int(0.8 * len(dataset))
test_length = length - train_length

train_data, test_data = random_split(dataset, (train_length, test_length))

