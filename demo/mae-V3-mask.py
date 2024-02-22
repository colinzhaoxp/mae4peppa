import sys
import os

import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('..')

import models

from engines.engine_test import evaluate
from util.create_dataset import create_dataset
from util.my_utils import load_train_configs, load_checkpoint

from tqdm import tqdm

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    # plt.title(title, fontsize=16)
    plt.axis('off')


def before_forward(img, depth_img):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # d_x = torch.tensor(depth_img)
    # d_x = d_x.unsqueeze(dim=0)
    # d_x = torch.einsum('nhwc->nchw', d_x)

    # return x, d_x

    return x, None

def run_one_image(img, model, save_dir=None):
    img, depth_img = img

    x, dep_x = before_forward(img, depth_img)

    # run MAE
    y = model((x.float(), None))
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


def load_img(img_url, depth_img_url):
    # load an image
    img = Image.open(img_url)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    # # load an depth image
    # depth_img = Image.open(depth_img_url)
    # depth_img = depth_img.resize((224, 224))
    # depth_img = np.array(depth_img) / 255.
    # depth_img = np.expand_dims(depth_img, axis=-1)
    #
    # assert depth_img.shape == (224, 224, 1)
    #
    # # normalize by ImageNet mean and std
    # depth_img = depth_img / 1.0
    #
    # return img, depth_img

    return img


def get_dataset(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.in_chans == 4:
        depth_norm = (0.005261, 0.011198)
        mean.append(depth_norm[0])
        std.append(depth_norm[1])

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=3),  # 3 is bicubic
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train, dataset_val, dataset_test = create_dataset(args.data_name, args, transform_train)

    return dataset_train, dataset_val, dataset_test


def main(args):

    print("==========\nArgs:{}\n==========".format(args))

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val, dataset_test = get_dataset(args)

    if dataset_test is None:
        dataset_test = dataset_val

    # define the model
    model = models.create(name=args.model, img_size=args.input_size, in_chans=args.in_chans, class_num=args.class_num,
                          patch_size=args.patch_size)

    print("start restore test dataset images")
    model = load_checkpoint(model, os.path.join(args.output_dir, 'checkpoint-best.pth'))

    # model.to(device)
    model.eval()

    img_save_dir = os.path.join(args.output_dir, "train_mask_recon")
    os.makedirs(img_save_dir, exist_ok=True)

    for (origin_rgb_path, origin_depth_path, weight, mask_rgb_path, mask_depth_path) in tqdm(dataset_test.dataset):
        # img, depth_img = load_img(mask_rgb_path, mask_depth_path)
        img = load_img(mask_rgb_path, mask_depth_path)
        save_dir = os.path.join(img_save_dir, os.path.basename(mask_rgb_path))
        # run_one_image((img, depth_img), model, save_dir)
        run_one_image((img, None), model, save_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="weight model test for V3 mask dataset")
    parser.add_argument("--config_file", default='/home/zhaoxp/workspace/mae-test/output_dir/2-14/configs.yaml')
    parser.add_argument('--data_name', default='peppa2depthV3', type=str, help='dataset name')
    parser.add_argument('--data_path', default='./data/peppa2depthV3', type=str,
                        help='dataset path')
    args = parser.parse_args()

    config_args = load_train_configs(args.config_file)
    config_args.data_name = args.data_name
    config_args.data_path = args.data_path

    main(config_args)

