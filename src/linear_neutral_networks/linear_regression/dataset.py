import torch
import pandas as pd
from torch.utils.data import random_split

DATA_PATH = "/home/chillmaw/Documents/pochodnia/torch_projects/src/linear_neutral_networks/linear_regression/data/housing.csv"

class HousingDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None):
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.df = self.df.dropna().reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        ocean_proximity_dict = {
            "NEAR OCEAN": 0,
            "ISLAND": 1,
            "INLAND": 2,
            "NEAR BAY": 3,
            "<1H OCEAN": 4
        }

        row = self.df.iloc[index]

        y = torch.tensor(
            [row["median_house_value"]],
            dtype=torch.float32
        )

        x = torch.tensor([
            row["longitude"],
            row["latitude"],
            row["housing_median_age"],
            row["total_rooms"],
            row["total_bedrooms"],
            row["population"],
            row["households"],
            row["median_income"]
        ], dtype=torch.float32)

        one_hot = torch.zeros(5, dtype=torch.float32)
        proximity = ocean_proximity_dict[row["ocean_proximity"]]
        one_hot[proximity] = 1.0

        x = torch.cat([x, one_hot])
        return x, y
    
dataset = HousingDataset(DATA_PATH)
dataset_length = len(dataset)



train_length = int(dataset_length * 0.8)
test_length = dataset_length - train_length
train_data, test_data = random_split(dataset, [train_length, test_length])

numeric_cols = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]

train_rows = dataset.df.iloc[train_data.indices]
means = train_rows[numeric_cols].mean()
stds = train_rows[numeric_cols].std().replace(0, 1)

dataset.df[numeric_cols] = (dataset.df[numeric_cols] - means) / stds