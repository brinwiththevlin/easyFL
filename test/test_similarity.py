import pytest
import torch
from src.similarity.plain_text import cosine_sim, kernel_sim, sim_matrix, graph_selector


def test_cosine_sim():
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    assert pytest.approx(cosine_sim(tensor1, tensor2), abs=1e-5) == 1.0

    tensor2 = torch.tensor([3, 2, 1])
    assert pytest.approx(cosine_sim(tensor1, tensor2), abs=1e-5) == 0.7142857313156128


def test_kernel_sim():
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    assert pytest.approx(kernel_sim(tensor1, tensor2), abs=1e-5) == 1.0

    tensor2 = torch.tensor([3, 2, 1])
    assert pytest.approx(kernel_sim(tensor1, tensor2), abs=1e-5) == 0.3678794503211975


def test_sim_matrix():
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    tensor3 = torch.tensor([3, 2, 1])
    weights_list = [tensor1, tensor2, tensor3]
    expected_matrix = [
        [1.0, 0.7142857313156128, 0.7142857313156128],
        [0.7142857313156128, 1.0, 0.3678794503211975],
        [0.7142857313156128, 0.3678794503211975, 1.0],
    ]
    assert (
        pytest.approx(sim_matrix(weights_list, cosine_sim), abs=1e-5) == expected_matrix
    )


def test_graph_selector():
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([1, 2, 3])
    tensor3 = torch.tensor([3, 2, 1])
    tensor4 = torch.tensor([3, 2, 1])
    tensor5 = torch.tensor([1, 2, 3])
    weights_list = [tensor1, tensor2, tensor3, tensor4, tensor5]
    expected_selector = [0, 1, 2]
    assert graph_selector(weights_list, 3, 0.65) == expected_selector
