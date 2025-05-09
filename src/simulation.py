# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning
import click
from typing import Iterable
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import VisionDataset
from config import load_config, Config
from similarity.plain_text import kmeans_selector
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
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="bad_filter.log",  # Log file destination
    filemode="a",  # Overwrite the log file each time, use "a" for append
)


class DatasetSplit(Dataset):
    def __init__(self, dataset: VisionDataset, idxs: Iterable[int]):
        # self.dataset = copy.deepcopy(dataset)
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item: int):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def label_tampering(self, config):
        if config.label_tampering == "zero":
            self.dataset.targets[self.idxs] = 0
        elif config.label_tampering == "reverse":
            self.dataset.targets[self.idxs] = 9 - self.dataset.targets[self.idxs]
        elif config.label_tampering == "random":
            self.dataset.targets[self.idxs] = torch.randint(0, 10, (len(self.idxs),))
        elif config.label_tampering == "none":
            self.dataset.targets[self.idxs] = self.dataset.targets[self.idxs]
        else:
            raise Exception("Unknown label tampering method name")


@click.command()
@click.option("--iterations", default=1000, help="number of global iterations")
@click.option("--iid", is_flag=True, help="set to true if the data is to be iid")
@click.option("--clients", default=100, help="total number of clients")
@click.option("--per_round", default=10, help="clints to select per round")
@click.option(
    "--selection",
    default="kl-kmeans",
    type=click.Choice(["kl-kmeans", "random"]),
)
@click.option("--bad_nodes", default=0, help="number of bad nodes")
@click.option("--res_path", default=None, help="path to save results")
@click.option(
    "--under_rep", type=int, default=3, help="number of under-represented classes"
)
@click.option("--dataset", default="MNIST", help="dataset to use")
@click.option(
    "--label_tampering",
    type=click.Choice(["none", "zero", "reverse", "random"]),
    default="none",
    help="label tampering",
)
@click.option(
    "--weight_tampering",
    type=click.Choice(["none", "large_neg", "reverse", "random"]),
    default="none",
    help="weight tampering",
)
def main(
    iterations: int,
    iid: bool,
    clients: int | None,
    per_round: int,
    selection: str,
    res_path: str | None,
    under_rep: int | None,
    dataset: str,
    label_tampering: str,
    weight_tampering: str,
    bad_nodes: int = 0,
) -> None:
    config = Config()
    config.parse_args(
        iterations,
        iid,
        clients,
        per_round,
        selection,
        res_path,
        under_rep,
        dataset,
        label_tampering,
        weight_tampering,
    )
    config.save_config()
    config = load_config()

    step_size = config.step_size

    random.seed(config.seed)
    np.random.seed(config.seed)  # numpy
    torch.manual_seed(config.seed)  # cpu
    torch.cuda.manual_seed(config.seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn

    ctx = click.get_current_context()
    arg_str = " ".join(f"{k}={v}" for k, v in ctx.params.items())
    print("Arguments received: " + arg_str)
    if clients is not None and per_round > clients:
        raise ValueError("per_round can't be higher the thotal number of clients")

    logging.info(f"Arguments received: {arg_str}")

    bad_subset = random.sample(range(clients), clients // 10)

    data_train, data_test, data_validate = load_data(
        config.dataset, config.dataset_file_path, config.model_name
    )
    data_train_loader = DataLoader(
        data_train, batch_size=config.batch_size_eval, shuffle=True, num_workers=0
    )  # num_workers=8
    data_test_loader = DataLoader(
        data_test, batch_size=config.batch_size_eval, shuffle=True, num_workers=0
    )  # num_workers=8
    # data_validation_loader = DataLoader(
    #     data_validate, batch_size=config.batch_size_eval, num_workers=0
    # )
    dict_users = split_data(config.dataset, data_train, config.n_nodes, config.iid)

    if config.n_nodes is None:
        config.n_nodes = len(dict_users)

    model: Models = get_model(
        config.model_name,
        config.dataset,
        rand_seed=config.seed,
        step_size=step_size,
        device=config.device,
        flatten_weight=config.flatten_weight,
    )

    stat = CollectStatistics(results_file_name=config.fl_results_file_path)
    train_loader_list: list[DataLoader] = []
    dataiter_list = []
    weight_list: list[torch.Tensor | dict] = [
        torch.empty(0) for _ in range(config.n_nodes)
    ]
    for n in range(config.n_nodes):
        data: DatasetSplit = DatasetSplit(data_train, dict_users[n])
        if n in bad_subset:
            data.label_tampering(config)
        train_loader_list.append(
            DataLoader(
                data,
                batch_size=config.batch_size_train,
                shuffle=True,
            )
        )
        dataiter_list.append(iter(train_loader_list[n]))

    w_global_init: torch.Tensor | dict[str, torch.Tensor] = model.get_weight()
    w_global = copy.deepcopy(w_global_init)

    num_iter = 0
    last_output = 0

    while True:
        w_global_prev = copy.deepcopy(w_global)

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
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(), max_norm=1.0
                )  # Prevent divergence
                model.optimizer.step()

            if n in bad_subset:
                if config.weight_tampering == "large_neg":
                    model.tamper_weights_large_negative()
                elif config.weight_tampering == "reverse":
                    model.tamper_weights_reverse()
                elif config.weight_tampering == "random":
                    model.tamper_weights_random()
                elif config.weight_tampering == "none":
                    pass
                else:
                    raise Exception("Unknown weight tampering method name")
            w = model.get_weight()  # deepcopy is already included here
            weight_list[n] = w


        if config.selection == "random":
            node_subset = np.random.choice(
                range(config.n_nodes), config.n_nodes_in_each_round, replace=False
            ).tolist()
            if num_iter % config.num_iter_one_output == 0:
                logging.info(
                    f"selection at iteration {num_iter}: random selection: {node_subset}"
                )
                if set(node_subset) & set(bad_subset):
                    logging.info(
                        f"***{set(node_subset) & set(bad_subset)} nodes passed the filter***"
                    )
        else:
            node_subset = kmeans_selector(
                model,
                weight_list,  # type: ignore
                train_loader_list,
                data_validate.targets,  # type: ignore
                bad_subset,
                label_tampering,
                weight_tampering,
                size=config.n_nodes_in_each_round,
                # tolerance=config.tolerance,
            )
            # node_subset = graph_selector(
            #     weight_list,  # type: ignore
            #     config.n_nodes_in_each_round,
            #     config.tolerance,
            # )

        for n in node_subset:
            if w_accu is None:  # accumulated weights
                w_accu = weight_list[n]
            else:
                if config.flatten_weight:
                    assert isinstance(w_accu, torch.Tensor)
                    w_accu += weight_list[n]
                else:
                    assert isinstance(w_accu, dict)
                    assert isinstance(weight_list[n], dict)
                    for k in w_accu.keys():
                        w_accu[k] += weight_list[n][k]

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
            model.scheduler.step()

        if num_iter >= config.max_iter:
            break

    stat.collect_stat_end()
    generate_figures()


if __name__ == "__main__":
    main()
    sys.exit(0)
