# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

from typing import Tuple, List, Dict, Any, Union
# %%
class SiameseNetworkTripletLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.remove_fc()

    def remove_fc(self):
        self.base = nn.Sequential(*list(self.base.children())[:-1])

    def forward(self, x):
        x =  self.base(x)
        return x.view(x.shape[0], x.shape[1])

# %%
# model = SiameseNetworkTripletLoss()
# # %%
# from Datasets import CelebADataset
# ds = CelebADataset()
# positive, _, _, _, _, _ = ds.__getitem__(0)
# # %%
# x = model(positive.unsqueeze(0))

# %%
