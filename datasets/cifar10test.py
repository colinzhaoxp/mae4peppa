import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from util.datasets import build_transform


def build_cifar_dataset(args, transform=None):
    use_depth = None
    if args.in_chans == 4:
        use_depth = (0.005261, 0.011198)

    if transform is None:
        transform = build_transform(False, use_depth, args)
    val_transform = build_transform(False, use_depth, args)

    training_dataset = datasets.CIFAR10(
        root="/home/zhaoxp/workspace/mae-test/data/",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="/home/zhaoxp/workspace/mae-test/data/",
        train=False,
        download=True,
        transform=val_transform,
    )

    return training_dataset, test_dataset, test_dataset


if __name__ == '__main__':

    train, val, test = build_minist_dataset(None)
    a = 1