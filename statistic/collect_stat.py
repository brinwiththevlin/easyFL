from torch.utils.data import DataLoader
from models.models import Models
from tensorboardX import SummaryWriter
from config import config
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__) + "/results.csv"):
        self.results_file_name = results_file_name
        self.summary_write = SummaryWriter(
            log_dir=os.path.join(config.results_file_path, config.comments)
        )
        self.summary_write_train = SummaryWriter(
            log_dir=os.path.join(config.results_file_path, config.comments, "train")
        )
        self.summary_write_test = SummaryWriter(
            log_dir=os.path.join(config.results_file_path, config.comments, "test")
        )

        with open(results_file_name, "a") as f:
            f.write("num_iter,lossValue,trainAccuracy,predictionAccuracy\n")
            f.close()

    def collect_stat_global(
        self,
        num_iter: int,
        model: Models,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        w_global=None,
    ):
        loss_value, train_accuracy = model.accuracy(
            train_data_loader, w_global, config.device
        )
        _, prediction_accuracy = model.accuracy(
            test_data_loader, w_global, config.device
        )
        prediction_precision = model.precision(
            test_data_loader, w_global, config.device
        )
        prediction_recall = model.recall(test_data_loader, w_global, config.device)
        prediction_f1 = (
            prediction_precision
            * prediction_recall
            * 2
            / (prediction_precision + prediction_recall)
        )

        self.summary_write.add_scalar("Loss", loss_value, num_iter)
        self.summary_write_train.add_scalar("Accuracy", train_accuracy, num_iter)
        self.summary_write_test.add_scalar("Accuracy", prediction_accuracy, num_iter)
        self.summary_write_test.add_scalar("Precision", prediction_precision, num_iter)
        self.summary_write_test.add_scalar("Recall", prediction_recall, num_iter)
        self.summary_write_test.add_scalar("F1", prediction_f1, num_iter)
        print(
            "Iter. "
            + str(num_iter)
            + "  train accu "
            + str(train_accuracy)
            + "  test accu "
            + str(prediction_accuracy)
            + "  test precision "
            + str(prediction_precision)
            + "  test recall "
            + str(prediction_recall)
            + "  test F1 "
            + str(prediction_f1)
        )

        with open(self.results_file_name, "a") as f:
            f.write(
                str(num_iter)
                + ","
                + str(loss_value)
                + ","
                + str(train_accuracy)
                + ","
                + str(prediction_accuracy)
                + ","
                + str(prediction_precision)
                + ","
                + str(prediction_recall)
                + ","
                + str(prediction_f1)
                + "\n"
            )
            f.close()

    def collect_stat_end(self):
        self.summary_write.close()
        self.summary_write_train.close()
        self.summary_write_test.close()


# tensorboard --logdir results
# tensorboard --logdir results --host=127.0.0.1
