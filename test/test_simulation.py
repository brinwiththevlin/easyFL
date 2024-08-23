import torch
import random
import networkx as nx


def sim_matrix(weights_list):
    """
    Compute the similarity matrix of the weights of the nodes
    """
    matrix = torch.zeros(len(weights_list), len(weights_list))
    for i, w1 in enumerate(weights_list):
        for j, w2 in enumerate(weights_list):
            matrix[i, j] = torch.dot(
                w1, w2) / (torch.norm(w1) * torch.norm(w2))
    return matrix


def graph_selector(wieghts_list, size, tolerance):
    matrix = sim_matrix(wieghts_list)
    G = nx.Graph()
    G.add_nodes_from(list(range(10)))
    sized_components = []
    for i in range(10):
        for j in range(i, 10):
            if matrix[i, j] > tolerance:  # TODO: check if > is correct
                G.add_edge(i, j)

    for gn in nx.connected_components(G):
        if len(gn) >= size:
            sized_components.append(gn)

    if len(sized_components) != 0:
        return random.choice(sized_components)

    # TODO: fix this line
    return graph_selector(sim_matrix, size, tolerance * 0.9)


def test_graph_selector():
    torch.random.manual_seed(0)
    weights_list = [torch.rand(10) * 10 for _ in range(10)]
    print(weights_list)
    graph = graph_selector(weights_list, 5, 0.5)
    assert len(graph) == 5
    print(graph)


def test_sim_matrix():
    torch.random.manual_seed(0)
    weights_list = [torch.rand(10) * 10 for _ in range(10)]
    print(weights_list)
    matrix = sim_matrix(weights_list)
    assert matrix.shape == (10, 10)
    assert torch.allclose(matrix, matrix.T)
    assert torch.allclose(torch.diag(matrix), torch.ones(10))
    print(matrix)


if __name__ == "__main__":
    test_sim_matrix()
    test_graph_selector()
    print("All tests passed")
