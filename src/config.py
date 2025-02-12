import os
import torch
import click
import yaml


class Config:
    def __init__(self):
        self.use_gpu = True
        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        # self.dataset = "MNIST"
        # self.model_name = "ModelCNNMnist"
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
        self.dataset_file_path = os.path.join(os.path.dirname(__file__), "dataset_data_files")
        self.results_file_path = os.path.join(os.path.dirname(__file__), "results/")
        self.fl_results_file_path = ""
        # ----------------------settings for clients
        self.n_nodes: int | None = 100  # None for fmnist and celeba, set a number for others
        self.n_nodes_in_each_round = 50
        self.step_size = 0.01  # learning rate of clients
        self.batch_size_train = 32
        self.batch_size_eval = 256
        self.max_iter = 1000  # Maximum number of iterations to run
        self.seed = 12345
        self.aggregation_method = "FedAvg"
        self.flatten_weight = True
        # self.iid = True
        self.iid = False
        self.tau_setup = 10  # number of iterations in local training
        self.num_iter_one_output = 50
        self.tolerance = None
        self.selection = "cosine"
        self.under_represented_classes = 0

    def set_results_file_path(self, res_path: str | None) -> None:
        self.comments = (
            self.dataset
            + "_"
            + ("iid" if self.iid else "noniid")
            + "_"
            + (f"{self.under_represented_classes}_" if not self.iid else "")
            + str(self.n_nodes)
            + "_"
            + str(self.n_nodes_in_each_round)
            + "_"
            + self.selection
        )

        if res_path:
            self.results_file_path = os.path.join(self.results_file_path, res_path)
        self.fl_results_file_path = os.path.join(self.results_file_path, self.comments, "results.csv")

    def parse_args(
        self,
        iterations: int,
        iid: bool,
        clients: int | None,
        per_round: int,
        selection: str,
        res_path: str | None,
        under_rep: int = 3,
        dataset: str = "mnist",
        label_tampering: str = "none",
        weight_tampering: str = "none",
    ):
        self.max_iter = iterations
        self.iid = iid
        self.n_nodes = clients
        self.n_nodes_in_each_round = per_round
        self.selection = selection
        self.dataset = dataset
        self.label_tampering = label_tampering
        self.weight_tampering = weight_tampering
        self.under_represented_classes = under_rep
        match dataset:
            case "MNIST":
                self.model_name = "ModelCNNMnist"
            case "cifar10":
                self.model_name = "ResNet34"
            case "cifar100":
                self.model_name = "ResNet34"
            case "SVHN":
                self.model_name = "WResNet40-2"
            case "FEMNIST":
                self.model_name = "ModelCNNEmnist"
        self.set_results_file_path(res_path)

    def save_config(self, filepath='config.yaml'):
        """Save configuration to a YAML file."""
        # Create a copy of the config dict with device converted to string
        config_dict = self.__dict__.copy()
        config_dict['device'] = str(self.device)  # Convert device to string
        with open(filepath, 'w') as file:
            yaml.safe_dump(config_dict, file)
        

def load_config(filepath='config.yaml'):
    """Load configuration from a YAML file."""
    with open(filepath, 'r') as file:
        config_dict = yaml.safe_load(file)
        config = Config()
        for key, value in config_dict.items():
            if key == 'device':
                value = torch.device(value)  # Convert string back to device
            setattr(config, key, value)
        return config

@click.command()
@click.option("--iterations", default=1000, help="number of global iterations")
@click.option("--iid", is_flag=True, help="set to true if the data is to be iid")
@click.option("--clients", default=100, help="total number of clients")
@click.option("--per_round", default=10, help="clints to select per round")
@click.option(
    "--selection",
    default="kl-kmeans",
    type=click.Choice(["kl-kmeans","random"]),
)
@click.option("--res_path", default=None, help="path to save results")
@click.option("--under_rep", type=int, default=3, help="number of under-represented classes")
@click.option("--dataset", default="MNIST", help="dataset to use")
@click.option("--bad_nodes", default=0, help="number of bad nodes")
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
    bad_nodes: int,
    dataset: str,
    label_tampering: str,
    weight_tampering: str,
) -> None:
    config = Config()
    config.parse_args(
        iterations, iid, clients, per_round, selection, res_path, under_rep, dataset, label_tampering, weight_tampering
    )
    config.save_config()
    # print(config.__dict__)
    # print(config.results_file_path)
    # print(config.fl_results_file_path)
    # print(config.device)

if __name__ == "__main__":
    main()