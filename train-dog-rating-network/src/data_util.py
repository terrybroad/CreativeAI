# This code is adapted from: https://stackoverflow.com/a/65232039

import os
import PIL
import torch
import numpy as np
import pandas as pd

class WeRateDogsDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform = None):
        csv_path = os.path.join(path, 'doggo_ratings.tsv')
        ims_path = os.path.join(path, 'images')
        self.df = pd.read_csv(csv_path, sep='\t')
        self.images_folder = ims_path
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def scale_and_to_tensor(self, rating):
        rating = rating / 16
        rating = np.clip(rating, 0, 1)
        rating = torch.tensor(rating, dtype=torch.float32)
        return rating

    def __getitem__(self, index):
        doggo_index = self.df.iloc[index]["doggo_index"]
        filename = f'{doggo_index:05}.jpg'
        rating = self.df.iloc[index]["doggo_rating"]
        rating = self.scale_and_to_tensor(rating)
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, rating