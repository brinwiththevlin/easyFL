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

    # return temp1.dot(temp2)
    # enc_v1 = ts.ckks_vector(context, temp1)
    # enc_v2 = ts.ckks_vector(context, temp2)

    # return enc_v1.dot(enc_v2).decrypt()[0]


def sum_of_squares(vector1: Tensor, vector2: Tensor) -> float:
    return torch.sum((vector1 - vector2) ** 2).item()
    # enc_v2 = ts.ckks_vector(context, vector2)
    # enc_v1 = ts.ckks_vector(context, vector1)
    #
    # return (enc_v1 - enc_v2).square().sum().decrypt()[0]


sim_functions = {"cosine": cosine_sim, "sos": sum_of_squares}


def sim_matrix(
    weights_list: list[Tensor], similarity: Callable[[Tensor, Tensor], float]
) -> np.ndarray:
    assert config.n_nodes is not None
    matrix = np.eye(config.n_nodes)
    for pair in combinations(enumerate(weights_list), 2):
        idx: list[int] = [x[0] for x in pair]
        values: list[Tensor] = [x[1] for x in pair]
        matrix[idx] = similarity(*values)
    return matrix
    # matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    # # difference between max value and min value not counting diagonal, which should be 0
    # diff = np.max(matrix) - \
    #     np.min(matrix[~np.eye(matrix.shape[0], dtype=bool)])
    # norm_matrix = matrix / diff
    # return norm_matrix


def graph_selector(
    weights_list: list[Tensor], size: int, tolerance: float
) -> list[int]:
    assert isinstance(
        weights_list[0], Tensor
    ), "not implemented for flattenweight = false"
    assert config.n_nodes is not None
    matrix: np.ndarray = sim_matrix(weights_list, sim_functions[config.similarity])
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
        sample_percent = max(
            floor((len(component) / config.n_nodes) * size), 1
        )  # always sample at least 1
        nodes.extend(random.sample(component, sample_percent))
    return nodes
