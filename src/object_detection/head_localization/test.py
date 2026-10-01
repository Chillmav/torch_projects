import pandas as pd
import torch

# Dataframe operations:

df = pd.read_csv("C:/python/torch_projects/src/object_detection/head_localization/data/annotations.csv")

head = df.columns.to_list()

num_columns = ['width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']
means = df[num_columns].mean()
stds = df[num_columns].std()


df[num_columns] = (df[num_columns] - means) / (stds + 1e-6)