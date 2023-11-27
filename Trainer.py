# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Datasets import CelebADataset
from SiameseModels import SiameseNetworkTripletLoss

from tqdm import tqdm
import numpy as np

from typing import Tuple, List, Dict, Any, Union
# %%
class Trainer:
    loader = None
    def __init__(self, model, root, train_batch_size, validation_batch_size, epochs) -> None:
        self.model = model
        self.root = root
        self.epoch = epochs
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size

        self.set_dataloaders()
        self.set_optimizer()
        self.set_loss()
        self.set_dicts()
        self.set_device()

    def set_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_dataloaders(self) -> None:
        self.train_loader = DataLoader(CelebADataset(root=self.root, train=True), batch_size=self.train_batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(CelebADataset(root=self.root, train=False), batch_size=self.validation_batch_size, shuffle=False, num_workers=4)

    def set_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def set_loss(self) -> None:
        self.distance_fn = nn.PairwiseDistance()
        self.loss_fn = nn.TripletMarginWithDistanceLoss(margin=1.0, distance_function=self.distance_fn)

    def set_dicts(self) -> None:
        self.step_dict = {"positive": [], "negative": [], "loss": []}
        self.trainer_dict = {"train": self.step_dict.copy(), "val": self.step_dict.copy()}

    def calculate_distance(self, anchor_out, positive_out, negative_out) -> Tuple:
        anchor_out = anchor_out.detach()
        positive_out = positive_out.detach()
        negative_out = negative_out.detach()
        
        positive_distance = self.distance_fn(anchor_out, positive_out).mean().item()
        negative_distance = self.distance_fn(anchor_out, negative_out).mean().item()   
        return positive_distance, negative_distance
    
    def switch_loader(self, train: bool) -> None:
        if train:
            self.model.train()
            self.loader = self.train_loader
        else:
            self.model.eval()
            self.loader = self.val_loader

    def set_pbar(self, pbar, tmp_dict) -> None:
        loss = np.mean(tmp_dict["loss"])
        positive = np.mean(tmp_dict["positive"])
        negative = np.mean(tmp_dict["negative"])
        pbar.set_description(f"Loss: {loss:.4f} | Positive: {positive:.4f} | Negative: {negative:.4f}")



    
    def train_step(self, train=False, epoch=None):
        self.switch_loader(train)
        pbar = tqdm(self.loader)
        tmp_dict = self.step_dict.copy()
        for anchor, anchor_id, positive, positive_id, negative, negative_id in pbar:
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            if train:
                self.optimizer.zero_grad()



            anchor_out = self.model(anchor)

            positive_out = self.model(positive)
            negative_out = self.model(negative)

            loss = self.loss_fn(anchor_out, positive_out, negative_out)

            if train:
                loss.backward()
                self.optimizer.step()

            tmp_dict["loss"].append(loss.item())
            positive_distance, negative_distance = self.calculate_distance(anchor_out, positive_out, negative_out)
            tmp_dict["positive"].append(positive_distance)
            tmp_dict["negative"].append(negative_distance)
            self.set_pbar(pbar, tmp_dict)

        loss = np.mean(tmp_dict["loss"])
        positive = np.mean(tmp_dict["positive"])
        negative = np.mean(tmp_dict["negative"])


        if train:
            self.trainer_dict["train"]["loss"].append(loss)
            self.trainer_dict["train"]["positive"].append(positive)
            self.trainer_dict["train"]["negative"].append(negative)
        else:
            self.trainer_dict["val"]["loss"].append(loss)
            self.trainer_dict["val"]["positive"].append(positive)
            self.trainer_dict["val"]["negative"].append(negative)

        return loss, positive, negative
    

    def train(self) -> None:
        min_loss = np.inf
        for epoch in range(self.epoch):
            self.train_step(train=True, epoch=epoch)
            loss, positive, negative = self.train_step(train=False, epoch=epoch)
            if loss < min_loss:
                min_loss = loss
                model_info = f"model_e_{epoch}_l_{loss}.pth"
                torch.save(self.model.state_dict(), model_info)
                print(f"Model saved as {model_info}")

        with open("trainer_dict.npy", "wb") as f:
            np.save(f, self.trainer_dict)







# %%
if __name__ == "__main__":
    trainer = Trainer(SiameseNetworkTripletLoss(), root="", train_batch_size=32, validation_batch_size=32, epochs=5)
    trainer.train()
# %%
