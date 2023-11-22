# %%
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import os 
import tqdm
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict, Any, Union
# %%
class CelebADataset(Dataset):
    def __init__(self, root: str = "", train: bool = True) -> None:
        self.root = root
        self.train = train
        self.set_df()
        self.set_transforms()


    def set_df(self):
        self.df = pd.read_csv(os.path.join(self.root, "data", "identity_CelebA.csv"))
        self.df = self.df[self.df["train"] == self.train]
        self.df.reset_index(drop=True, inplace=True)

    def set_transforms(self):
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        

    def __len__(self) -> int:
        return self.df.shape[0]
    

    def get_image(self, idx: int) -> Image:
        image_id = self.df.iloc[idx, 0]
        image = Image.open(os.path.join(self.root, "data", "img", image_id))
        image = self.transform(image)
        return image
    

    def __getitem__(self, idx) -> Tuple:
        anchor_idx = self.df.index[idx]
        anchor_id = self.df.iloc[idx, 1]
        positive_idx = self.df[self.df["id"] == anchor_id].sample(1).index[0]
        negative_idx = self.df[self.df["id"] != anchor_id].sample(1).index[0]

        anchor = self.get_image(anchor_idx)
        positive = self.get_image(positive_idx)
        negative = self.get_image(negative_idx)

        return anchor, anchor_id, positive, self.df.loc[positive_idx, "id"], negative, self.df.loc[negative_idx, "id"]

