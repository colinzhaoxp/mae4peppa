import os
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm

recon_rgb_path = "/home/zhaoxp/workspace/mae-test/output_dir/2-14-2/train_mask_recon/"
mask_rgb_path = "/home/zhaoxp/workspace/mae-test/data/peppa2depthV3/rgb_mask/"

save_path = "/home/zhaoxp/workspace/mae-test/output_dir/2-14-2/train_mask_recon_concat/"


for file_path in tqdm(os.listdir(recon_rgb_path)):
    mask_rgb = Image.open(osp.join(mask_rgb_path, file_path))
    recon_rgb = Image.open(osp.join(recon_rgb_path, file_path))

    # rgb中的黑色部分作为可以被覆盖的部分
    mask = Image.eval(mask_rgb, lambda x: 0 if x > 100 else 255).convert("L")

    mask_rgb.paste(recon_rgb, (0, 0), mask)

    mask_rgb.save(osp.join(save_path, file_path))
