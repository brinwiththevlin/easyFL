import copy
from typing import Any, Sequence
from torch.autograd import Variable
import torch
from torch.optim.sgd import SGD
from torch import device, nn
from torch.functional import Tensor
from torch.utils.data import DataLoader
from models.wresnet import WideResNet
from models.lenet import LeNet5
from models.resnet import ResNet34, ResNet18
from functools import reduce
import collections
import os
import sys
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    MultiplicativeLR,
    ConstantLR,
    ReduceLROnPlateau,
)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE
LOSS_ACC_BATCH_SIZE = 128


class Models:
    def __init__(
        self,
        rand_seed: int | None = None,
        learning_rate: float = 0.001,
        num_classes: int = 10,
        model_name: str = "LeNet5",
        channels: int = 1,
        img_size: int = 32,
        device: torch.device | None = None,
        flatten_weight: bool = False,
    ):
        super(Models, self).__init__()
        if device is None:
            device = torch.device("cuda")

        if rand_seed is not None:
            _ = torch.manual_seed(rand_seed)
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        self.channels = channels
        self.img_size = img_size
        self.flatten_weight = flatten_weight
        self.learning_rate = learning_rate
        # self.model: nn.Module | None = None

        if model_name == "ModelCNNMnist":
            from models.cnn_mnist import ModelCNNMnist

            self.model = ModelCNNMnist().to(device)
            self.init_variables()
        elif model_name == "ModelCNNEmnist":
            from models.cnn_emnist import ModelCNNEmnist

            self.model = ModelCNNEmnist().to(device)
            self.init_variables()
        elif model_name == "ModelCNNEmnistLeaf":
            from models.cnn_emnist_leaf import ModelCNNEmnist

            self.model = ModelCNNEmnist().to(device)
        elif model_name == "ModelCNNCifar10":
            from models.cnn_cifar10 import ModelCNNCifar10

            self.model = ModelCNNCifar10().to(device)
            self.init_variables()
        elif model_name == "ModelCNNCeleba":
            from models.cnn_celeba import ModelCNNCeleba

            self.model = ModelCNNCeleba().to(device)
        elif model_name == "LeNet5":
            self.model = LeNet5()
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # lr 0.001
        elif model_name == "ResNet34":
            self.model = ResNet34(num_classes=num_classes)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)  # lr 0.1 adjustable
        elif model_name == "ResNet18":
            self.model = ResNet18(num_classes=num_classes)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        elif model_name == "WResNet40-2":
            self.model = WideResNet(
                depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif model_name == "WResNet16-1":
            self.model = WideResNet(
                depth=16, num_classes=num_classes, widen_factor=1, dropRate=0.0
            )
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"model_name {model_name} is not supported")

        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scheduler = MultiplicativeLR(self.optimizer, lr_lambda=0.1)
        _ = self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self._get_weight_info()

    def get_first_layer_weight(self) -> Tensor:
        with torch.no_grad():
            state = self.model.state_dict()
            return state[self.weights_key_list[0]].view(-1)

    def weight_variable(self, tensor: Tensor, mean: float, std: float) -> Tensor:
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        _ = tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        _ = tensor.data.mul_(std).add_(mean)
        return tensor

    def bias_variable(self, shape: Sequence[int]) -> Tensor:
        return torch.ones(shape) * 0.1

    def init_variables(self) -> None:
        self._get_weight_info()

        weight_dic = collections.OrderedDict()

        for i in range(len(self.weights_key_list)):
            if i % 2 == 0:
                tensor = torch.zeros(self.weights_size_list[i])
                sub_weight = self.weight_variable(tensor, 0, 0.1)
            else:
                sub_weight = self.bias_variable(self.weights_size_list[i])
            weight_dic[self.weights_key_list[i]] = sub_weight

        self.model.load_state_dict(weight_dic)

    def _get_weight_info(self) -> None:
        self.weights_key_list = []
        self.weights_size_list = []
        self.weights_num_list = []
        state = self.model.state_dict()
        for k, v in state.items():
            shape = list(v.size())
            self.weights_key_list.append(k)
            self.weights_size_list.append(shape)
            if len(shape) > 0:
                num_w = reduce(lambda x, y: x * y, shape)
            else:
                num_w = 0
            self.weights_num_list.append(num_w)

    def get_weight_dimension(self) -> int:
        dim = sum(self.weights_num_list)
        return dim

    def get_weight(self) -> Tensor | dict[str, Tensor]:
        with torch.no_grad():
            state = self.model.state_dict()
            if self.flatten_weight:
                weight_flatten_tensor = torch.Tensor(sum(self.weights_num_list)).to(
                    state[self.weights_key_list[0]].device
                )
                start_index = 0
                for i, [_, v] in zip(range(len(self.weights_num_list)), state.items()):
                    weight_flatten_tensor[
                        start_index : start_index + self.weights_num_list[i]
                    ] = v.view(1, -1)
                    start_index += self.weights_num_list[i]

                return weight_flatten_tensor
            else:
                return copy.deepcopy(state)

    def tamper_weights_large_negative(self) -> None:
        tampered_state = {
            k: torch.ones_like(v) * -10 for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(tampered_state)

    def tamper_weights_reverse(self) -> None:
        tampered_state = {
            k: torch.ones_like(v) * -1 for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(tampered_state)

    def tamper_weights_random(self) -> None:
        tampered_state = {
            k: (torch.rand_like(v) * 100) - 50
            for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(tampered_state)

    def assign_weight(self, w: Tensor | dict[str, Any]) -> None:
        if self.flatten_weight:
            assert isinstance(w, Tensor)
            self.assign_flattened_weight(w)
        else:
            assert isinstance(w, dict)
            self.model.load_state_dict(w)

    def assign_flattened_weight(self, w: Tensor) -> None:
        weight_dic = collections.OrderedDict()
        start_index = 0

        for i in range(len(self.weights_key_list)):
            sub_weight = w[start_index : start_index + self.weights_num_list[i]]
            if len(sub_weight) > 0:
                weight_dic[self.weights_key_list[i]] = sub_weight.view(
                    self.weights_size_list[i]
                )
            else:
                weight_dic[self.weights_key_list[i]] = torch.tensor(0)
            start_index += self.weights_num_list[i]
        self.model.load_state_dict(weight_dic)

    def _data_reshape(
        self, imgs: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if len(imgs.size()) < 3:
            x_image = imgs.view([-1, self.channels, self.img_size, self.img_size])
            if labels is not None:
                # From one-hot to number
                _, y_label = torch.max(labels.data, 1)
            else:
                y_label = None
            return x_image, y_label
        else:
            return imgs, labels

    def gradient(
        self,
        imgs: Tensor,
        labels: Tensor,
        w: Tensor | dict[str, Tensor],
        sampleIndices: Sequence[int] | None,
        device: torch.device,
    ) -> Tensor | dict[str, Tensor]:
        self.assign_weight(w)

        if sampleIndices is None:
            sampleIndices = range(0, len(labels))

        imgs = imgs[sampleIndices].to(device)
        labels = labels[sampleIndices].to(device)

        imgs, labels = self._data_reshape(imgs, labels)  # type: ignore

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(imgs)
        loss = self.loss_fn(output, labels)
        loss.backward()

        return self.get_weight()

    def accuracy(
        self,
        data_test_loader: DataLoader,
        w: Tensor | dict[str, Tensor] | None,
        device: device,
    ) -> tuple[float, float]:
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        total_correct = 0
        avg_loss: Tensor = torch.empty((1,)).to(device)
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = (
                    Variable(images).to(device),
                    Variable(labels).to(device),
                )
                output = self.model(images)
                avg_loss += self.loss_fn(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        avg_loss /= len(data_test_loader.dataset)  # type: ignore
        acc = float(total_correct) / len(data_test_loader.dataset)  # type: ignore

        return round(avg_loss.item(), 5), round(acc, 5)

    def precision(
        self,
        data_test_loader: DataLoader,
        w: Tensor | dict[str, Tensor] | None,
        device: torch.device,
    ) -> float:
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        true_positive = 0
        false_positive = 0

        with torch.no_grad():
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = self.model(images)
                pred = output.data.max(1)[1]  # Get the predicted class

                # Calculate the true positives (correctly classified samples)
                true_positive += (pred == labels).sum().item()

                # Calculate the false positives (incorrectly classified samples)
                false_positive += (pred != labels).sum().item()

        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            precision = 0.0

        return round(precision, 5)

    def recall(
        self,
        data_test_loader: DataLoader,
        w: Tensor | dict[str, Tensor] | None,
        device: torch.device,
    ) -> float:
        if w is not None:
            self.assign_weight(w)

        self.model.eval()
        true_positive = 0
        total_actual_positive = 0

        with torch.no_grad():
            for _, (images, labels) in enumerate(data_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = self.model(images)
                pred = output.data.max(1)[1]  # Get the predicted class

                # Calculate the true positives (correctly classified samples)
                true_positive += (pred == labels).sum().item()

                # Total actual positives (total number of samples)
                total_actual_positive += labels.size(0)

        try:
            recall = true_positive / total_actual_positive
        except ZeroDivisionError:
            recall = 0.0

        return round(recall, 5)

    def f1(
        self,
        data_test_loader: DataLoader,
        w: Tensor | dict[str, Tensor] | None,
        device: torch.device,
    ) -> float:
        precision = self.precision(data_test_loader, w, device)
        recall = self.recall(data_test_loader, w, device)
        try:
            f1 = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return round(f1, 5)

    def predict(
        self, img: Tensor, w: Tensor | dict[str, Tensor], device: torch.device
    ) -> Tensor:
        self.assign_weight(w)
        img, _ = self._data_reshape(img)
        with torch.no_grad():
            self.model.eval()
            _, pred = torch.max(self.model(img.to(device)).data, 1)

        return pred

    def train_one_epoch(
        self, data_train_loader: DataLoader, device: torch.device
    ) -> None:
        self.model.train()
        for _, (images, labels) in enumerate(data_train_loader):
            images, labels = Variable(images).to(device), Variable(labels).to(device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step(device)
