# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning
import click
from typing import Iterable
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import VisionDataset
from config import config
from similarity.plain_text import graph_selector
from datasets.dataset import load_data
from models.get_model import get_model
from models.models import Models
from statistic.collect_stat import CollectStatistics
from statistic.figure import generate_figures
from util.sampling import split_data
import numpy as np
import random
import copy
import sys


class DatasetSplit(Dataset):
    def __init__(self, dataset: VisionDataset, idxs: Iterable[int]):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item: int):
        image, label = self.dataset[self.idxs[item]]
        return image, label


@click.command()
@click.option("--iterations", default=1000, help="number of global iterations")
@click.option("--iid", is_flag=True, help="set to true if the data is to be iid")
@click.option("--clients", default=10, help="total number of clients")
@click.option("--per_round", default=5, help="clints to select per round")
@click.option(
    "--selection",
    default="cosine",
    type=click.Choice(["cosine", "pearson", "kernel", "random"]),
)
@click.option("--res_path", default=None, help="path to save results")
@click.option("--tolerance", type=float, default=None, help="tolerance for similarity")
def main(
    iterations: int,
    iid: bool,
    clients: int | None,
    per_round: int,
    selection: str,
    res_path: str | None,
    tolerance: float | None,
) -> None:
    print(
        f"Arguments received: iterations={iterations}, iid={iid}, clients={clients}, per_round={per_round}, selection={selection}, res_path={res_path}"
    )
    if clients is not None and per_round > clients:
        raise ValueError("per_round can't be higher the thotal number of clients")

    config.parse_args(
        iterations, iid, clients, per_round, selection, res_path, tolerance
    )
    random.seed(config.seed)
    np.random.seed(config.seed)  # numpy
    torch.manual_seed(config.seed)  # cpu
    torch.cuda.manual_seed(config.seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn

    data_train, data_test = load_data(
        config.dataset, config.dataset_file_path, config.model_name
    )
    data_train_loader = DataLoader(
        data_train, batch_size=config.batch_size_eval, shuffle=True, num_workers=0
    )  # num_workers=8
    data_test_loader = DataLoader(
        data_test, batch_size=config.batch_size_eval, num_workers=0
    )  # num_workers=8
    dict_users = split_data(config.dataset, data_train, config.n_nodes, config.iid)

    if config.n_nodes is None:
        config.n_nodes = len(dict_users)

    model: Models = get_model(
        config.model_name,
        config.dataset,
        rand_seed=config.seed,
        step_size=config.step_size,
        device=config.device,
        flatten_weight=config.flatten_weight,
    )

    stat = CollectStatistics(results_file_name=config.fl_results_file_path)
    train_loader_list = []
    dataiter_list = []
    weight_list: list[torch.Tensor | dict] = [
        torch.empty(0) for _ in range(config.n_nodes)
    ]
    for n in range(config.n_nodes):
        train_loader_list.append(
            DataLoader(
                DatasetSplit(data_train, dict_users[n]),
                batch_size=config.batch_size_train,
                shuffle=True,
            )
        )
        dataiter_list.append(iter(train_loader_list[n]))

    w_global_init: torch.Tensor | dict[str, torch.Tensor] = model.get_weight()
    w_global = copy.deepcopy(w_global_init)

    num_iter = 0
    last_output = 0

    first = True
    while True:
        w_global_prev = copy.deepcopy(w_global)

        if config.selection == "random":
            node_subset = np.random.choice(
                range(config.n_nodes), config.n_nodes_in_each_round, replace=False
            )
        else:
            if first:
                node_subset = np.random.choice(
                    range(config.n_nodes), config.n_nodes_in_each_round, replace=False
                )
                first = False
            else:
                node_subset = graph_selector(
                    weight_list,  # type: ignore
                    config.n_nodes_in_each_round,
                    config.tolerance,
                )

        w_accu = None
        for n in range(config.n_nodes):
            model.assign_weight(w_global)
            # model.train_one_epoch(train_loader_list[n], device)
            model.model.train()
            for _ in range(0, config.tau_setup):
                try:
                    images, labels = next(dataiter_list[n])
                    if len(images) < config.batch_size_train:
                        # dataiter_list[n] = iter(train_loader_list[n])
                        images, labels = next(dataiter_list[n])
                except StopIteration:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = next(dataiter_list[n])

                images, labels = images.to(config.device), labels.to(config.device)
                model.optimizer.zero_grad()
                output = model.model(images)
                loss = model.loss_fn(output, labels)
                loss.backward()
                model.optimizer.step()

            w = model.get_weight()  # deepcopy is already included here
            weight_list[n] = w

            if n in node_subset:
                if w_accu is None:  # accumulated weights
                    w_accu = w
                else:
                    if config.flatten_weight:
                        assert isinstance(w_accu, torch.Tensor)
                        w_accu += w
                    else:
                        assert isinstance(w_accu, dict)
                        assert isinstance(w, dict)
                        for k in w_accu.keys():
                            w_accu[k] += w[k]

        num_iter = num_iter + config.tau_setup

        if config.aggregation_method == "FedAvg":
            if config.flatten_weight:
                assert isinstance(w_accu, torch.Tensor)
                w_global = torch.div(
                    copy.deepcopy(w_accu),
                    torch.tensor(config.n_nodes_in_each_round).to(config.device),
                ).view(-1)
            else:
                assert isinstance(w_accu, dict)
                assert isinstance(w_global, dict)
                for k in w_global.keys():
                    w_global[k] = torch.div(  # type: ignore
                        copy.deepcopy(w_accu[k]),
                        torch.tensor(config.n_nodes_in_each_round).to(config.device),
                    )
        else:
            raise Exception("Unknown parameter server method name")

        has_nan = False
        if config.flatten_weight:
            assert isinstance(w_global, torch.Tensor)
            if (True in torch.isnan(w_global)) or (True in torch.isinf(w_global)):
                has_nan = True
        else:
            assert isinstance(w_global, dict)
            for k in w_global.keys():
                if (True in torch.isnan(w_global[k])) or (  # type: ignore
                    True in torch.isinf(w_global[k])  # type: ignore
                ):  # type: ignore
                    has_nan = True
        if has_nan:
            print("*** w_global is NaN or InF, using previous value")
            w_global = copy.deepcopy(w_global_prev)

        if num_iter - last_output >= config.num_iter_one_output:
            stat.collect_stat_global(
                num_iter, model, data_train_loader, data_test_loader, w_global
            )
            last_output = num_iter

        if num_iter >= config.max_iter:
            break

    stat.collect_stat_end()
    generate_figures()


if __name__ == "__main__":
    main()
    sys.exit(0)
