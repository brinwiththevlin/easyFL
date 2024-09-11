from math import floor
from typing import Callable, List
import torch
import numpy as np
import networkx as nx
import random
from itertools import combinations

from torch.functional import Tensor
from config import config


def cosine_sim(vector1: Tensor, vector2: Tensor) -> float:
    temp1 = vector1 / torch.norm(vector1)
    temp2 = vector2 / torch.norm(vector2)

    return temp1.dot(temp2).item()

    # return temp1.dot(temp2)
    # enc_v1 = ts.ckks_vector(context, temp1)
    # enc_v2 = ts.ckks_vector(context, temp2)

    # return enc_v1.dot(enc_v2).decrypt()[0]


def sum_of_squares(vector1: Tensor | dict, vector2: Tensor | dict) -> float:
    assert isinstance(vector1, Tensor), "not implemented for dict"
    assert isinstance(vector2, Tensor), "not implemented for dict"
    return torch.sum((vector1 - vector2) ** 2).item()
    # enc_v2 = ts.ckks_vector(context, vector2)
    # enc_v1 = ts.ckks_vector(context, vector1)
    #
    # return (enc_v1 - enc_v2).square().sum().decrypt()[0]


sim_functions = {"cosine": cosine_sim, "sos": sum_of_squares}


def sim_matrix(weights_list: List[Tensor | dict], similarity: Callable) -> np.ndarray:
    matrix = np.eye(config.n_nodes)
    for pair in combinations(enumerate(weights_list), 2):
        idx, values = zip(*pair)
        matrix[idx] = similarity(*values)
    return matrix
    # matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    # # difference between max value and min value not counting diagonal, which should be 0
    # diff = np.max(matrix) - \
    #     np.min(matrix[~np.eye(matrix.shape[0], dtype=bool)])
    # norm_matrix = matrix / diff
    # return norm_matrix


def graph_selector(weights_list: List[Tensor | dict], size: int, tolerance: float) -> List[int]:
    matrix = sim_matrix(weights_list, sim_functions[config.similarity])
    G = nx.Graph()
    G.add_nodes_from(list(range(config.n_nodes)))

    for i in range(config.n_nodes):
        for j in range(i, config.n_nodes):
            if matrix[i, j] > tolerance:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    components = [list(c) for c in components]
    nodes = []
    for component in components:
        sample_percent = max(floor((len(component)/config.n_nodes) * size), 1) # always sample at least 1
        nodes.extend(random.sample(component, sample_percent))
    return nodes
