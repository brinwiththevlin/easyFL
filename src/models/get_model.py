from datasets.dataset import get_data_info
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_model(
    model_name: str,
    dataset: str,
    rand_seed: int | None = None,
    step_size: float = 0.001,
    device: torch.device = torch.device("cuda"),
    flatten_weight: bool = False,
):
    img_size, channels, num_classes = get_data_info(dataset, model_name)

    from models.models import Models

    return Models(
        rand_seed,
        step_size,
        num_classes=num_classes,
        model_name=model_name,
        channels=channels,
        img_size=img_size,
        device=device,
        flatten_weight=flatten_weight,
    )
