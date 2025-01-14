#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, SVHN
from torchvision.datasets.mnist import MNIST

from datasets.celeba import CelebA
from datasets.femnist import FEMNIST


def mnist_iid(dataset: MNIST, num_users: int) -> dict[int, set[int]]:
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users: dict[int, set[int]] = {}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset: MNIST, num_users: int):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    labels = dataset.targets.numpy()
    num_classes = len(set(labels))
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}


    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dominant_classes = [np.random.choice(num_classes, 2, replace=True) for _ in range(num_users)]

    dominant_ratio = 0.7

    num_samples_per_user = len(dataset) // num_users

    for user_id in range(num_users):
        dominant_class_pair = dominant_classes[user_id]
        num_dominant_samples = int(dominant_ratio * num_samples_per_user)

        dominant_class_data = [np.random.choice(
            class_indices[dominant_class],
            size=num_dominant_samples//2,
        ) for dominant_class in dominant_class_pair]
        dict_users[user_id] = np.concatenate((dict_users[user_id], dominant_class_data[0],dominant_class_data[1]))

        remaining_classes = [c for c in range(num_classes) if c not in dominant_class_pair]
        num_other_samples = num_samples_per_user - num_dominant_samples

        for class_id in remaining_classes:
            num_class_samples = int(num_other_samples / (num_classes - 1))

            class_data = np.random.choice(
                class_indices[class_id],
                size=num_class_samples,
                replace=True  # Allow ov
            )
            dict_users[user_id] = np.concatenate((dict_users[user_id], class_data))

    return dict_users


def cifar_iid(dataset: CIFAR10, num_users: int) -> dict[int, set[int]]:
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users: dict[int, set[int]] = {}
    all_idxs: list[int] = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset: CIFAR10, num_users: int) -> dict[int, np.ndarray]:
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    labels = np.array(dataset.targets)
    num_classes = len(set(labels))
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}


    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dominant_classes = [np.random.choice(num_classes, 2, replace=True) for _ in range(num_users)]

    dominant_ratio = 0.7

    num_samples_per_user = len(dataset) // num_users

    for user_id in range(num_users):
        dominant_class_pair = dominant_classes[user_id]
        num_dominant_samples = int(dominant_ratio * num_samples_per_user)

        dominant_class_data = [np.random.choice(
            class_indices[dominant_class],
            size=num_dominant_samples//2,
        ) for dominant_class in dominant_class_pair]
        dict_users[user_id] = np.concatenate((dict_users[user_id], dominant_class_data[0],dominant_class_data[1]))

        remaining_classes = [c for c in range(num_classes) if c not in dominant_class_pair]
        num_other_samples = num_samples_per_user - num_dominant_samples

        for class_id in remaining_classes:
            num_class_samples = int(num_other_samples / (num_classes - 1))

            class_data = np.random.choice(
                class_indices[class_id],
                size=num_class_samples,
                replace=True  # Allow ov
            )
            dict_users[user_id] = np.concatenate((dict_users[user_id], class_data))

    return dict_users
    # num_shards, num_imgs = (
    #     2 * num_users,
    #     int(len(dataset.data) / 2 / num_users),
    # )  # choose two number from a set with num_shards, each client has 2*num_imgs images
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
    # idxs = np.arange(len(dataset.data))
    # labels = np.array(dataset.targets)
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    #
    # # divide and assign
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
    #         )
    # return dict_users


def svhn_iid(dataset: SVHN, num_users: int):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def svhn_noniid(dataset: SVHN, num_users: int):
    num_shards, num_imgs = (
        2 * num_users,
        int(len(dataset.data) / 2 / num_users),
    )  # choose two number from a set with num_shards, each client has 2*num_imgs images
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype="int64") for i in range(num_users)}
    idxs = np.arange(len(dataset.data))
    labels = dataset.labels

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def split_data(
    dataset_name: str,
    data_train: datasets.VisionDataset,
    n_nodes: int | None,
    iid: bool = False,
):
    if dataset_name == "MNIST":
        if n_nodes is None:
            raise Exception("Unknown n_nodes for MNIST.")
        if iid:
            assert isinstance(data_train, MNIST)
            dict_users = mnist_iid(data_train, n_nodes)
        else:
            assert isinstance(data_train, MNIST)
            dict_users = mnist_noniid(data_train, n_nodes)
    elif dataset_name == "cifar10" or dataset_name == "cifar100":
        if n_nodes is None:
            raise Exception("Unknown n_nodes for CIFAR*.")
        if iid:
            assert isinstance(data_train, CIFAR10)
            dict_users = cifar_iid(data_train, n_nodes)
        else:
            assert isinstance(data_train, CIFAR10)
            dict_users = cifar_noniid(data_train, n_nodes)
    elif dataset_name == "FEMNIST":
        if iid:
            assert isinstance(data_train, FEMNIST)
            raise Exception("Only consider NON-IID setting in FEMNIST")
        else:
            assert isinstance(data_train, FEMNIST)
            dict_users = data_train.get_dict_clients()

    elif dataset_name == "celeba":
        if iid:
            raise Exception("Only consider NON-IID setting in FEMNIST")
        else:
            assert isinstance(data_train, CelebA)
            dict_users = data_train.get_dict_clients()
    elif dataset_name == "SVHN":
        if n_nodes is None:
            raise Exception("Unknown n_nodes for SVHN.")
        if iid:
            assert isinstance(data_train, SVHN)
            dict_users = svhn_iid(data_train, n_nodes)
        else:
            assert isinstance(data_train, SVHN)
            dict_users = svhn_noniid(data_train, n_nodes)
    else:
        raise Exception("Unknown dataset name.")
    return dict_users


if __name__ == "__main__":
    dataset_train = datasets.MNIST(
        "../data/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    num = 100
    d = mnist_noniid(dataset_train, num)
