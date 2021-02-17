import os
import csv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, parameters):
        self.log_path = parameters.log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.loss_history = []
        self.acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.parameters = parameters

    def collect_history(self, loss, accuracy, val_loss, val_accuracy):
        self.loss_history.append(loss)
        self.acc_history.append(accuracy)
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_accuracy)

    def draw_graph(self):
        plt.plot(self.loss_history, label="loss")
        plt.plot(self.val_loss_history, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(self.log_path, "loss.png"))
        plt.gca().clear()
        plt.plot(self.acc_history, label="accuracy")
        plt.plot(self.val_acc_history, label="val_accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.log_path, "accuracy.png"))

    def save_parameters_and_result(self):
        last_train_loss = self.loss_history[len(self.loss_history) - 1]
        last_train_acc = self.acc_history[len(self.acc_history) - 1]
        last_val_loss = self.val_loss_history[len(self.val_loss_history) - 1]
        last_val_acc = self.val_acc_history[len(self.val_acc_history) - 1]
        dict = {"Train Acc": last_train_acc.item(), "Val Acc": last_val_acc.item(),
                "Train Loss": last_train_loss, "Val Loss": last_val_loss,
                "Dataset": "MNIST",
                "Epochs": self.parameters.learning_parameters["epochs"],
                "Batch Size": self.parameters.learning_parameters["batch_size"],
                "Optimizer": "Adam",
                "Learning Rate": self.parameters.learning_parameters["learning_rate"],
                "Layers Num": self.parameters.model_parameters["layers_num"],
                "Heads": self.parameters.model_parameters["heads"],
                "Tokens": self.parameters.model_parameters["tokens"],
                "Dims": int(784//self.parameters.model_parameters["tokens"]),
                "Hidden Dims": self.parameters.model_parameters["hidden_dim"],
                "Dropout Rate": self.parameters.model_parameters["dropout_rate"]}
        with open(os.path.join(self.log_path, "parameters.csv"), "w") as f:
            writer = csv.DictWriter(f, list(dict.keys()))
            writer.writeheader()
            writer.writerow(dict)
