from tqdm import tqdm
import time
pbar = tqdm(range(10000))
for i in pbar:
    time.sleep(0.001)
    if i < 5000:
        pbar.set_description(f"Less than 5000")
    else:
        pbar.set_description(f"More than 5000")
