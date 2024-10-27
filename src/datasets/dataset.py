# Part of this code is adapted from https://github.com/jz9888/federated-learning
import os
import random
import sys

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
from torchvision.datasets.mnist import MNIST

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_data_info(dataset: str, model_name: str) -> tuple[int, int, int]:
    if dataset == "MNIST":
        if model_name == "ModelCNNMnist":
            img_size = 28
        elif model_name == "LeNet5":
            img_size = 32
        else:
            raise Exception("Unknown model name for MNIST.")
        channels = 1
        num_classes = 10
    elif dataset == "SVHN":
        img_size = 32
        channels = 3
        num_classes = 10
    elif dataset == "cifar10":
        img_size = 32
        channels = 3
        num_classes = 10
    elif dataset == "cifar100":
        img_size = 32
        channels = 3
        num_classes = 100
    elif dataset == "FEMNIST":
        img_size = 28
        channels = 1
        num_classes = 62
    elif dataset == "celeba":
        img_size = 84
        channels = 3
        num_classes = 2
    else:
        raise Exception("Unknown dataset name.")
    return img_size, channels, num_classes


def load_data(
    dataset: str, data_path: str, model_name: str
) -> tuple[VisionDataset, VisionDataset, VisionDataset]:
    """load test, train and validation data for the given dataset
    where test and validation datasets have the same label distribution
    Args:
        dataset (str): name of the dataset
        data_path (str): path to the dataset
        model_name (str): name of the model
    Returns:
        tuple: train, test and validation datasets
    """

    def unbalanced_classes(test_val_index, targets):
        targets = [targets[i] for i in test_val_index]
        under_represented_classes = random.sample(range(10), 3)
        class_limit = 500
        class_dist = [0] * 10
        new_test_val_index = []
        for i, idx in enumerate(test_val_index):
            if (
                targets[i] in under_represented_classes
                and class_dist[targets[i]] < class_limit
            ):
                new_test_val_index.append(idx)
                class_dist[targets[i]] += 1
            else:
                new_test_val_index.append(idx)

        return new_test_val_index

    # TODO: only works for MNIST
    def create_subset(dataset: MNIST, indices: list[int]) -> VisionDataset:
        subset_data = dataset.data[indices]
        subset_targets = dataset.targets[indices]

        mnist_subset = MNIST(
            root=data_full.root,
            train=True,
            transform=data_full.transform,
            download=False,
        )
        mnist_subset.data = subset_data
        mnist_subset.targets = subset_targets

        return mnist_subset

    img_size, _, _ = get_data_info(dataset, model_name)
    if dataset == "MNIST":
        # Load MNIST dataset
        data_full = MNIST(
            data_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
            download=True,
        )
        # Perform train-validation split
        train_idx, test_val_idx = train_test_split(
            range(len(data_full)),
            test_size=0.4,
            stratify=data_full.targets,
        )

        test_val_idx = unbalanced_classes(test_val_idx, data_full.targets)
        test_idx, val_idx = train_test_split(
            test_val_idx,
            test_size=0.5,
            stratify=[data_full.targets[i] for i in test_val_idx],
        )

        data_train = create_subset(data_full, train_idx)
        data_test = create_subset(data_full, test_idx)
        data_val = create_subset(data_full, val_idx)

    # Keep the rest of the code unchanged for other datasets
    elif dataset == "SVHN":
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        transform_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        data_train = SVHN(
            data_path + "/SVHN", split="train", download=True, transform=transform_train
        )
        data_test = SVHN(
            data_path + "/SVHN", split="test", download=True, transform=transform_test
        )
        data_val = SVHN(
            data_path + "/SVHN", split="test", download=True, transform=transform_test
        )

    elif dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        data_train = CIFAR10(
            data_path, transform=transform_train, download=True
        )  # True for the first time
        data_test = CIFAR10(data_path, train=False, transform=transform_test)
        data_val = CIFAR10(data_path, train=False, transform=transform_test)

    elif dataset == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        data_train = CIFAR100(
            data_path, transform=transform_train, download=True
        )  # True for the first time
        data_test = CIFAR100(data_path, train=False, transform=transform_test)
        data_val = CIFAR100(data_path, train=False, transform=transform_test)

    elif dataset == "FEMNIST":
        from datasets.femnist import FEMNIST

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        data_train = FEMNIST(data_path + "/femnist/", transform=transform, train=True)
        data_test = FEMNIST(data_path + "/femnist/", transform=transform, train=False)
        data_val = FEMNIST(data_path + "/femnist/", transform=transform, train=False)

    elif dataset == "celeba":
        from datasets.celeba import CelebA

        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [
                transforms.CenterCrop((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        data_train = CelebA(
            data_path + "/celeba/",
            transform=transform,
            train=True,
            read_all_data_to_mem=False,
        )
        data_test = CelebA(
            data_path + "/celeba/",
            transform=transform,
            train=False,
            read_all_data_to_mem=False,
        )
        data_val = CelebA(
            data_path + "/celeba/",
            transform=transform,
            train=False,
            read_all_data_to_mem=False,
        )

    else:
        raise Exception("Unknown dataset name.")

    return data_train, data_test, data_val
