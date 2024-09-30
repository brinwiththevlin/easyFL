from math import floor
import torch
import numpy as np
import networkx as nx
import random
from itertools import combinations
from typing import Callable

from torch import Tensor
from config import config


def cosine_sim(vector1: Tensor, vector2: Tensor) -> float:
    norm1: float = vector1.norm().item()
    norm2: float = vector2.norm().item()
    # Normalize the vectors
    temp1: Tensor = vector1 / norm1
    temp2: Tensor = vector2 / norm2

    # Compute the dot product and return it as a float
    dot_product = float(
        temp1.dot(temp2).item()
    )  # Ensure it returns a scalar as a float
    return float(dot_product)


def pearson_correlation(vector1: Tensor, vector2: Tensor) -> float:
    #r =\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum\left(x_{i}-\bar{x}\right)^{2} \sum\left(y_{i}-\bar{y}\right)^{2}}}
    mean1 = vector1.mean().item()
    mean2 = vector2.mean().item()
    diff1 = vector1 - mean1
    diff2 = vector2 - mean2
    numerator = torch.dot(diff1, diff2).item()
    denominator = torch.sqrt(torch.dot(diff1, diff1) * torch.dot(diff2, diff2)).item()
    return numerator / denominator

def kernel_sim(vector1: Tensor, vector2: Tensor, gamma: float | None= None) -> float:
    """uses radial baisis kernel (gaussian)"""
    if gamma is None:
        gamma = 1.0 / len(vector1)  # Default gamma
    diff = vector1 - vector2
    return torch.exp(-gamma * torch.dot(diff, diff)).item()


sim_functions = {"cosine": cosine_sim, "pearson": pearson_correlation, "kernel": kernel_sim}


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


def graph_selector(
    weights_list: list[Tensor], size: int, tolerance: float
) -> list[int]:
    assert isinstance(
        weights_list[0], Tensor
    ), "not implemented for flattenweight = false"
    assert config.n_nodes is not None
    matrix: np.ndarray = sim_matrix(weights_list, sim_functions[config.selection])

    # using matrix create a partition where node [n] = { X \subset N | \forall i,j \in X, matrix[i,n] > tolerance matrix[j,n] > tolerance matrix[i,j] > tolerance}
    partition = []
    found =  False
    for i in range(config.n_nodes):
        for part in partition:
            if all(matrix[i,j] > tolerance for j in part):
                part.append(i)
                found = True
                break

        if not found:
            partition.append([i])

        found = False

    nodes = []
    for part in partition:
        sample_percent = max(
            floor((len(part) / config.n_nodes) * size), 1
        )  # always sample at least 1
        nodes.extend(random.sample(part, sample_percent))

    return nodes



    # G = nx.Graph()
    # G.add_nodes_from(list(range(config.n_nodes)))
    #
    # for i in range(config.n_nodes):
    #     for j in range(i, config.n_nodes):
    #         if matrix[i, j] > tolerance:
    #             G.add_edge(i, j)
    #
    # components = list(nx.connected_components(G))
    # components = [list(c) for c in components]
    # nodes: list[int] = []
    # for component in components:
    #     sample_percent = max(
    #         floor((len(component) / config.n_nodes) * size), 1
    #     )  # always sample at least 1
    #     nodes.extend(random.sample(component, sample_percent))
    # return nodes
