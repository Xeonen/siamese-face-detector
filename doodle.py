# %%
import torch
import torch.nn as nn
from torchvision import transforms

import pandas as pd
import numpy as np

from PIL import Image

from SiameseModels import SiameseNetworkTripletLoss

# %%
model = SiameseNetworkTripletLoss()
model.load_state_dict(torch.load("model_e_4_l_0.1467277131297434.pth"))
# %%
df = pd.read_csv("data/identity_CelebA.csv", sep=",", header=0)
df = df[df["train"] == False]
# %%
transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
# %%
ids = df.id.sample(10).values
tmp_df = df[df["id"].isin(ids)]
# %%
model = model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
img_dict = {}

for id in tmp_df.id.unique():
    img_df = tmp_df[tmp_df["id"] == id]
    size = img_df.shape[0]
    for i, image_id in enumerate(img_df.image_id):
        img = Image.open(f"data/img/{image_id}")
        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        out = model(img)
        out = out.detach().cpu().numpy()
        l = img_dict.get(id, [])
        l.append(out)
        img_dict[id] = l

# %%
distance_fn = nn.PairwiseDistance()
key_dict = {}
for key in img_dict.keys():
    key_dict[key] = np.mean(img_dict[key][:-2], axis=0)
# %%
for anchor_key in img_dict.keys():
    anchor = img_dict[anchor_key][-1]
    for reference_key in key_dict.keys():

        reference = key_dict[reference_key]

        print(distance_fn(torch.tensor(anchor), torch.tensor(reference)).item(), anchor_key, reference_key)

# %%
