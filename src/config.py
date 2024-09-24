from datetime import datetime
import os

import torch


class Config:
    def __init__(self):
        self.use_gpu = True
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.dataset = "MNIST"
        self.model_name = "ModelCNNMnist"
        # self.model_name = 'LeNet5'
        # self.dataset = 'cifar10'
        # self.model_name = 'ModelCNNCifar10'
        # self.model_name = 'ResNet34'
        # self.model_name = 'ResNet18'
        # self.dataset = 'cifar100'
        # self.model_name = 'ResNet34'
        # self.model_name = 'ResNet18'
        # self.dataset = 'SVHN'
        # self.model_name = 'WResNet40-2'
        # self.model_name = 'WResNet16-1'
        # self.dataset = 'FEMNIST'
        # self.model_name = 'ModelCNNEmnist'
        # self.model_name = 'ModelCNNEmnistLeaf'
        # self.dataset = 'celeba'
        # self.model_name = 'ModelCNNCeleba'
        self.dataset_file_path = os.path.join(
            os.path.dirname(__file__), "dataset_data_files"
        )
        self.results_file_path = os.path.join(os.path.dirname(__file__), "results/")
        self.fl_results_file_path = ""
        # ----------------------settings for clients
        self.n_nodes: int | None = (
            100  # None for fmnist and celeba, set a number for others
        )
        self.n_nodes_in_each_round = 50
        self.step_size = 0.01  # learning rate of clients
        self.batch_size_train = 32
        self.batch_size_eval = 256
        self.max_iter = 1000  # Maximum number of iterations to run
        self.seed = 1
        self.aggregation_method = "FedAvg"
        self.random_node_selection = True
        self.flatten_weight = True
        # self.iid = True
        self.iid = False
        self.tau_setup = 10  # number of iterations in local training
        self.num_iter_one_output = 50
        self.tolerance = 0.9995 if self.iid else 0.7
        self.similarity = "cosine"

    def set_results_file_path(self, res_path: str | None) -> None:
        self.comments = (
            self.dataset
            + "-"
            + self.model_name
            + "-"
            + ("iid" if self.iid else "noniid")
            + "-"
            + str(self.n_nodes)
            + "-"
            + str(self.n_nodes_in_each_round)
            + "-"
            + self.similarity
            + "-"
            + "-"
            + "random"
            if self.random_node_selection
            else "graph" + datetime.now().strftime("%m-%d-%H:%M")
        )
        if res_path:
            self.results_file_path = os.path.join(self.results_file_path, res_path)
        self.fl_results_file_path = os.path.join(
            self.results_file_path, self.comments, "results.csv"
        )

    def parse_args(
        self,
        iterations: int,
        iid: int,
        clients: int| None,
        per_round: int,
        similarity: str,
        selection: str,
        res_path: str | None,
        tolerance: float| None = None,
    ):
        config.max_iter = iterations
        config.iid = iid
        config.n_nodes = clients
        config.n_nodes_in_each_round = per_round
        config.similarity = similarity
        config.set_results_file_path(res_path)
        config.random_node_selection = selection == "random"
        if tolerance is not None:
            config.tolerance = tolerance


config = Config()
