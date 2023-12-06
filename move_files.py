import os
import shutil
import pandas as pd
import numpy as np

from tqdm import tqdm


df = pd.read_csv("data/identity_CelebA.csv", sep=",", header=0)
df = df[df["train"] == False]
root = "data"
for id in tqdm(df.id.unique()):
    img_df = df[df["id"] == id]
    id = str(id)
    loc = os.path.join(root, "val", id)
    if not os.path.exists(loc):
        os.makedirs(loc)
    for image_id in img_df.image_id:
        shutil.copy(os.path.join(root, "img", image_id), loc)    
