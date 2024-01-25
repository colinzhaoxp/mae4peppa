import sys
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('..')
from models import models_mae

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # plt.title(title, fontsize=16)
    plt.axis('off')

def run_one_image(img, model, save_dir=None):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    y = model(x.float())
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    y = torch.clip((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255)
    im = Image.fromarray(y.numpy().astype(np.uint8))
    im.save(save_dir)
    # y = torch.einsum('hwc->chw', y)
    # save_image(y, save_dir)

    # x = torch.einsum('nchw->nhwc', x)

    # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]

    # plt.subplot(1, 2, 1)
    # show_image(x[0], "original")

    # plt.subplot(1, 2, 2)
    # show_image(y[0], "reconstruction")

    # plt.savefig(save_dir)


def load_img(img_url):
    # load an image
    img = Image.open(img_url)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    return img

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


if __name__ == "__main__":
    model_path = "/home/zhaoxp/workspace/mae-test/output_dir/7/checkpoint-15.pth"
    models_mae = prepare_model(model_path)

    # img_dir = "/home/zhaoxp/workspace/mae-test/data/peppa/train/train_mask"
    img_dir = './'
    img_save_dir = "/home/zhaoxp/workspace/mae-test/data/peppa/train/train_mask_recon"
    os.makedirs(img_save_dir, exist_ok=True)

    cnt = 10

    for root, _, fpaths in os.walk(img_dir):
        for fpath in fpaths:
            if not fpath.endswith('jpg'): continue
            img_url = os.path.join(root, fpath)
            img = load_img(img_url)
            save_dir = os.path.join(img_save_dir, fpath)
            run_one_image(img, models_mae, save_dir)
            cnt -= 1
            if cnt <= 0:
                break