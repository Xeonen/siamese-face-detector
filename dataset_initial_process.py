"""
This script performs dataset inspection and preprocessing for Siamese Face Detection.
"""

import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm


# Load the dataset
loc = "F:\\Datasets\\CelebA"
df = pd.read_csv(os.path.join(loc, "Anno", "identity_CelebA.txt"), sep=" ", header=None)
# image_id: image file name, id: identity number of the celebrity
df.columns = ["image_id", "id"]

# Filter out celebrities with fewer than 30 images
threshold = 30
mask = (df.id.value_counts() >= threshold)
mask = mask[mask.values == True].index
df = df[df.id.isin(mask)]

# Split the dataset into train and validation sets
val_split = 0.15
df.sort_values("id", inplace=True)
df.reset_index(drop=True, inplace=True)
val_id = df.iloc[int((1-val_split)*df.shape[0]), 1]
df["train"] = df["id"].apply(lambda x: True if x < val_id else False)


# Copy the images to the target directory
def copy_file(df, loc, target, sub_dir = "img"):
    for file in tqdm(df["image_id"]):
        file = file.replace("jpg", "png")
        src = os.path.join(loc, "img", "img_align_celeba_png", file)
        dst = os.path.join(target, sub_dir, file)
        shutil.copy(src, dst)

def copy_grouped(df, loc):
    for id in tqdm(df["id"]):
        tmp_df = df.iloc[np.where(df["id"] == id)[0], :]
        if not os.path.exists(os.path.join("data", "grouped_img", str(id))):
            os.makedirs(os.path.join("data", "grouped_img", str(id)))
        copy_file(tmp_df, loc, "data", sub_dir = os.path.join("grouped_img", str(id)))
        


# Create the target directory for the preprocessed images
target = "data/img"
if not os.path.exists(target):
    os.makedirs(target)

# Copy the images to the target directory
copy_file(df, loc, "data", "img")

# Save the preprocessed dataset as a CSV file
df["image_id"] = df["image_id"].apply(lambda x: x.replace("jpg", "png"))
df.to_csv(os.path.join("data", "identity_CelebA.csv"), sep=",", index=False)


