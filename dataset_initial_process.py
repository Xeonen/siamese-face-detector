"""
This script performs dataset inspection and preprocessing for Siamese Face Detection.
"""

import pandas as pd
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

# Create the target directory for the preprocessed images
target = "data/img"
if not os.path.exists(target):
    os.makedirs(target)

# Copy the images to the target directory
for file in tqdm(df["image_id"]):
    file = file.replace("jpg", "png")
    src = os.path.join(loc, "img", "img_align_celeba_png", file)
    dst = os.path.join(target, file)
    shutil.copy(src, dst)

# Save the preprocessed dataset as a CSV file
df.to_csv(os.path.join("data", "identity_CelebA.csv"), sep=",", index=False)
