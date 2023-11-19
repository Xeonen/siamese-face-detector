# %%
import pandas as pd
import os
import shutil
from tqdm import tqdm
# %%
loc = "F:\\Datasets\\CelebA"

df = pd.read_csv(os.path.join(loc, "Anno", "identity_CelebA.txt"), sep=" ", header=None)
df.columns = ["image_id", "id"]
df.head()
# %%
threshold = 30
# We will be getting celebs with only 33 or more images
mask = (df.id.value_counts() >= threshold)
mask = mask[mask.values == True].index
df = df[df.id.isin(mask)]
print(df.shape)
# %%
target = "data/img"
if not os.path.exists(target):
    os.makedirs(target)
df.to_csv(os.path.join("data", "identity_CelebA.csv"), sep=",", index=False)

for file in tqdm(df["image_id"]):
    file = file.replace("jpg", "png")
    src = os.path.join(loc, "img", "img_align_celeba_png", file)
    dst = os.path.join(target, file)
    shutil.copy(src, dst)

# %%
df["id"].value_counts().hist(bins=100)
# %%
