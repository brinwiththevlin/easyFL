# Part of this code is inspired by https://github.com/IBM/adaptive-federated-learning

from itertools import combinations
from torch.utils.data import Dataset, DataLoader
import torch
from config import config
from datasets.dataset import load_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.sampling import split_data
import numpy as np
import networkx as nx
import random
import copy

random.seed(config.seed)
np.random.seed(config.seed)  # numpy
torch.manual_seed(config.seed)  # cpu
torch.cuda.manual_seed(config.seed)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

data_train, data_test = load_data(
    config.dataset, config.dataset_file_path, config.model_name)
data_train_loader = DataLoader(
    data_train, batch_size=config.batch_size_eval, shuffle=True, num_workers=0)  # num_workers=8
data_test_loader = DataLoader(
    data_test, batch_size=config.batch_size_eval, num_workers=0)  # num_workers=8
dict_users = split_data(config.dataset, data_train,
                        config.n_nodes, config.iid)

if config.n_nodes is None:
    config.n_nodes = len(dict_users)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def cosine_sim(vector1, vector2):
    temp1 = vector1 / torch.norm(vector1)
    temp2 = vector2 / torch.norm(vector2)

    return temp1.dot(temp2)

    # return temp1.dot(temp2)
    # enc_v1 = ts.ckks_vector(context, temp1)
    # enc_v2 = ts.ckks_vector(context, temp2)

    # return enc_v1.dot(enc_v2).decrypt()[0]


def sum_of_squares(vector1, vector2):
    return torch.sum((vector1 - vector2) ** 2)
    # enc_v2 = ts.ckks_vector(context, vector2)
    # enc_v1 = ts.ckks_vector(context, vector1)
    #
    # return (enc_v1 - enc_v2).square().sum().decrypt()[0]


def sim_matrix(weights_list):
    matrix = np.eye(config.n_nodes)
    for pair in combinations(enumerate(weights_list), 2):
        idx, values = zip(*pair)
        matrix[idx] = cosine_sim(*values)
    return matrix
    # matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    # # difference between max value and min value not counting diagonal, which should be 0
    # diff = np.max(matrix) - \
    #     np.min(matrix[~np.eye(matrix.shape[0], dtype=bool)])
    # norm_matrix = matrix / diff
    # return norm_matrix


def graph_selector(weights_list, size, tolerance):
    matrix = sim_matrix(weights_list)
    G = nx.Graph()
    G.add_nodes_from(list(range(config.n_nodes)))
    for i in range(config.n_nodes):
        for j in range(i, config.n_nodes):
            if matrix[i, j] >= tolerance:  # TODO: check if > is correct
                G.add_edge(i, j)
    # reject the largets connected component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    components.pop(0)
    # return remaining nodes as a list
    nodes = []
    for component in components:
        nodes.extend(component)
    
    if nodes != []:
        return nodes
    else:
        return graph_selector(weights_list, size, tolerance *1.1)

    # for gn in nx.connected_components(G):
    #     if len(gn) >= size:
    #         sized_components.append(gn)

    # if len(sized_components) != 0:
    #     return random.choice(sized_components)

    # # TODO: fix this line
    # return graph_selector(sim_matrix, size, tolerance * 0.9)


model = get_model(config.model_name, config.dataset, rand_seed=config.seed,
                  step_size=config.step_size, device=config.device, flatten_weight=config.flatten_weight)

stat = CollectStatistics(results_file_name=config.fl_results_file_path)
train_loader_list = []
dataiter_list = []
weight_list = [0] * config.n_nodes
for n in range(config.n_nodes):
    train_loader_list.append(DataLoader(DatasetSplit(
        data_train, dict_users[n]), batch_size=config.batch_size_train, shuffle=True))
    dataiter_list.append(iter(train_loader_list[n]))

w_global_init = model.get_weight()
w_global = copy.deepcopy(w_global_init)

num_iter = 0
last_output = 0

first = True
while True:
    w_global_prev = copy.deepcopy(w_global)

    if first:
        node_subset = range(config.n_nodes)
        first = False
    else:
        node_subset = graph_selector(
            weight_list, config.n_nodes_in_each_round, config.tolerance)

    # if config.random_node_selection:
    #     node_subset = np.random.choice(
    #         range(config.n_nodes), config.n_nodes_in_each_round, replace=False)
    # else:
    #     node_subset = range(0, config.n_nodes_in_each_round)

    w_accu = None
    for n in node_subset:
        model.assign_weight(w_global)
        # model.train_one_epoch(train_loader_list[n], device)
        model.model.train()
        for i in range(0, config.tau_setup):
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

        w = model.get_weight()   # deepcopy is already included here
        weight_list[n] = w

        if w_accu is None:  # accumulated weights
            w_accu = w
        else:
            if config.flatten_weight:
                w_accu += w
            else:
                for k in w_accu.keys():
                    w_accu[k] += w[k]

    num_iter = num_iter + config.tau_setup

    if config.aggregation_method == 'FedAvg':
        if config.flatten_weight:
            w_global = torch.div(copy.deepcopy(w_accu), torch.tensor(
                config.n_nodes_in_each_round).to(config.device)).view(-1)
        else:
            for k in w_global.keys():
                w_global[k] = torch.div(copy.deepcopy(
                    w_accu[k]), torch.tensor(config.n_nodes_in_each_round).to(config.device))
    else:
        raise Exception("Unknown parameter server method name")

    has_nan = False
    if config.flatten_weight:
        if (True in torch.isnan(w_global)) or (True in torch.isinf(w_global)):
            has_nan = True
    else:
        for k in w_global.keys():
            if (True in torch.isnan(w_global[k])) or (True in torch.isinf(w_global[k])):
                has_nan = True
    if has_nan:
        print('*** w_global is NaN or InF, using previous value')
        w_global = copy.deepcopy(w_global_prev)

    if num_iter - last_output >= config.num_iter_one_output:
        stat.collect_stat_global(
            num_iter, model, data_train_loader, data_test_loader, w_global)
        last_output = num_iter

    if num_iter >= config.max_iter:
        break

stat.collect_stat_end()
