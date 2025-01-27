from math import floor
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch
import torch.utils.data
import numpy as np
import networkx as nx
import random
from itertools import combinations
from typing import Callable

from torch import Tensor
from config import load_config, Config

config = load_config("config.yaml")


def cosine_sim(vector1: Tensor, vector2: Tensor) -> float:
    norm1: float = vector1.norm().item()
    norm2: float = vector2.norm().item()
    # Normalize the vectors
    temp1: Tensor = vector1 / norm1
    temp2: Tensor = vector2 / norm2

    # Compute the dot product and return it as a float
    dot_product = float(temp1.dot(temp2).item())  # Ensure it returns a scalar as a float
    return float(dot_product)


def pearson_correlation(vector1: Tensor, vector2: Tensor) -> float:
    # r =\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum\left(x_{i}-\bar{x}\right)^{2} \sum\left(y_{i}-\bar{y}\right)^{2}}}
    mean1 = vector1.mean().item()
    mean2 = vector2.mean().item()
    diff1 = vector1 - mean1
    diff2 = vector2 - mean2
    numerator = torch.dot(diff1, diff2).item()
    denominator = torch.sqrt(torch.dot(diff1, diff1) * torch.dot(diff2, diff2)).item()
    return numerator / denominator


def kernel_sim(vector1: Tensor, vector2: Tensor, gamma: float | None = None) -> float:
    """uses radial baisis kernel (gaussian)"""
    if gamma is None:
        gamma = 1.0 / len(vector1)  # Default gamma
    diff = vector1 - vector2
    return torch.exp(-gamma * torch.dot(diff, diff)).item()


sim_functions = {
    "cosine": cosine_sim,
    "pearson": pearson_correlation,
    "kernel": kernel_sim,
}


def sim_matrix(weights_list: list[Tensor], similarity: Callable) -> np.ndarray:
    assert config.n_nodes is not None
    matrix = np.eye(config.n_nodes)
    for pair in combinations(enumerate(weights_list), 2):
        idx: list[int] = [x[0] for x in pair]
        values: list[Tensor] = [x[1] for x in pair]
        sim_value = similarity(*values)
        matrix[idx[0], idx[1]] = sim_value
        matrix[idx[1], idx[0]] = sim_value

    return matrix


def graph_selector(weights_list: list[Tensor], size: int, tolerance: float) -> list[int]:
    assert isinstance(weights_list[0], Tensor), "not implemented for flattenweight = false"
    assert config.n_nodes is not None
    matrix: np.ndarray = sim_matrix(weights_list, sim_functions[config.selection])

    G = nx.Graph()
    G.add_nodes_from(list(range(config.n_nodes)))

    for i in range(config.n_nodes):
        for j in range(i, config.n_nodes):
            if matrix[i, j] > tolerance:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    components = [list(c) for c in components]
    nodes: list[int] = []
    for component in components:
        sample_percent = max(floor((len(component) / config.n_nodes) * size), 1)  # always sample at least 1
        nodes.extend(random.sample(component, sample_percent))
    return nodes


def kl_divergence(dataset1_targets, dataset2_targets, num_classes):
    if isinstance(dataset1_targets, list):
        dataset1_targets = torch.tensor(dataset1_targets).to(config.device)
    if isinstance(dataset2_targets, list):
        dataset2_targets = torch.tensor(dataset2_targets).to(config.device)
    # Count the occurrences of each class
    counts1 = torch.bincount(dataset1_targets, minlength=num_classes).to(config.device)
    counts2 = torch.bincount(dataset2_targets, minlength=num_classes).to(config.device)

    # Normalize to get the probabilities
    prob1 = counts1.float() / len(dataset1_targets)
    prob2 = counts2.float() / len(dataset2_targets)

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    prob1 += epsilon
    prob2 += epsilon

    # Calculate KL divergence
    kl_div = F.kl_div(prob1.log(), prob2, reduction="batchmean")
    return kl_div.item()


def kmeans_selector(
    weights_list: list[Tensor],
    train_loader_list: list[torch.utils.data.DataLoader],
    val_targets: Tensor,
    bad_subset: list[int],
    label_tampering: str,
    size: int,
    # tolerance: float,
) -> list[int]:
    # WARNING: only works for MNIST and CIFAR
    local_targets = []
    for i, loader in enumerate(train_loader_list):
        targets = []
        if i in bad_subset:
            if label_tampering == "random":
                for images, labels in loader:
                    targets.extend(torch.randint(0, 10, (len(labels),)))
            elif label_tampering == "reverse":
                for images, labels in loader:
                    targets.extend(9 - labels)
            elif label_tampering == "zero":
                for images, labels in loader:
                    targets.extend(torch.zeros_like(labels))
            elif label_tampering == "none":
                for images, labels in loader:
                    targets.extend(labels)
            else:
                raise Exception("Unknown label tampering method name")
            local_targets.append(torch.tensor(targets))
            continue
        for images, labels in loader:
            targets.extend(labels)
        local_targets.append(torch.tensor(targets))

    divergences = [kl_divergence(x, val_targets, 10) for x in local_targets]
    tolerance = get_tolerance(divergences)
    new_weights_list = [
        weights_list[i] if x < tolerance else torch.zeros_like(weights_list[i]) for i, x in enumerate(divergences)
    ]
    non_zero_indices = [i for i, v in enumerate(new_weights_list) if not torch.all(v == 0)]
    non_zero_vectors = torch.stack([new_weights_list[i] for i in non_zero_indices])
    # convert non_zero_vectors to numpy
    non_zero_vectors = non_zero_vectors.cpu().numpy()
    kmeans = KMeans(n_clusters=size)
    kmeans.fit(non_zero_vectors)
    # chose one example from each cluster closest to the center
    centers = kmeans.cluster_centers_
    # convet back to torch
    centers = torch.tensor(centers).to(config.device)
    non_zero_vectors = torch.tensor(non_zero_vectors).to(config.device)
    closest: list[int] = []
    for center in centers:
        # Calculate distances between the center and all non-zero vectors
        distances = torch.norm(non_zero_vectors - center, dim=1)
        # Get the index of the closest vector within the non-zero vectors
        closest_idx = int(torch.argmin(distances).item())
        # Map it back to the original weights_list indices
        closest.append(non_zero_indices[closest_idx])

    return closest


def get_tolerance(divergences: list[float]):
    return np.percentile(divergences, 85)
