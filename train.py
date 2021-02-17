import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from parameter_loader import Parameters
from model import TransformerClassification
from logger import Logger
from optimizer import NoamOpt
from dataloader import *



class Trainer:
    def __init__(self, setting_yaml_file):
        self.parameters = Parameters(setting_yaml_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  #GPUが利用可能であればGPUを利用

        self.dataloader = DataLoader(data_path=self.parameters.data_path,
                                        batch_size=self.parameters.learning_parameters["batch_size"],
                                        img_size=self.parameters.img_size,
                                        num_workers=self.parameters.fixed_parameters["num_workers"],
                                        pin_memory=self.parameters.fixed_parameters["pin_memory"])

        self.model = TransformerClassification(heads=self.parameters.model_parameters["heads"],
                                                layers_num=self.parameters.model_parameters["layers_num"],
                                                dropout_rate=self.parameters.model_parameters["dropout_rate"],
                                                d_model=(784//self.parameters.model_parameters["tokens"]),
                                                d_ff=self.parameters.model_parameters["hidden_dim"],
                                                max_seq_len=self.parameters.model_parameters["tokens"], output_dim=10)
        self.logger = Logger(self.parameters)                                                         #ログ書き込みを行うLoggerクラスの宣言

    def train(self):
        self.model.train()
        for i in range(self.parameters.model_parameters["layers_num"]):
            self.model.encoders[i].apply(self.weights_init)
        #self.model.net3_1.apply(self.weights_init)
        #self.model.net3_2.apply(self.weights_init)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.parameters.learning_parameters["learning_rate"])

        #torch.backends.cudnn.benchmark = True

        with tqdm(range(self.parameters.learning_parameters["epochs"])) as progress_bar:
            for epoch in enumerate(progress_bar):
                i = epoch[0]
                progress_bar.set_description("[Epoch %d]" % (i+1))
                epoch_loss = 0.
                epoch_corrects = 0
                val_loss_result = 0.0
                val_acc = 0.0

                self.model.train()
                j = 1
                for images, labels in self.dataloader.dataloaders["train"]:
                    progress_bar.set_description("[Epoch %d (Iteration %d)]" % ((i+1), j))
                    j = j + 1
                    inputs = images.to(self.device)
                    labels = labels.to(self.device)

                    input_pad = 1
                    input_mask = inputs#(inputs != input_pad)
                    input_mask = input_mask.to(self.device)

                    outputs = self.model(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

                else:
                    with torch.no_grad():
                        self.model.eval()
                        progress_bar.set_description("[Epoch %d (Validation)]" % (i+1))
                        for inputs, labels in self.dataloader.dataloaders["val"]:
                            inputs = inputs.to(self.device)
                            labels = labels.to(self.device)

                            input_pad = 1
                            input_mask = inputs#(inputs != input_pad)
                            input_mask = input_mask.to(self.device)

                            outputs = self.model(inputs, input_mask)
                            loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            val_loss_result += loss.item()
                            val_acc += torch.sum(preds == labels.data)

                    epoch_loss = epoch_loss / len(self.dataloader.dataloaders["train"].dataset)
                    epoch_acc = epoch_corrects.float() / len(self.dataloader.dataloaders["train"].dataset)
                    val_epoch_loss = val_loss_result / len(self.dataloader.dataloaders["val"].dataset)
                    val_epoch_acc = val_acc.float() / len(self.dataloader.dataloaders["val"].dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)
                    self.logger.writer.add_scalars("losses", {"train":epoch_loss,"validation":val_epoch_loss}, (i+1))
                    self.logger.writer.add_scalars("accuracies", {"train":epoch_acc, "validation":val_epoch_acc}, (i+1))

                progress_bar.set_postfix({"loss":epoch_loss, "accuracy": epoch_acc.item(), "val_loss":val_epoch_loss, "val_accuracy": val_epoch_acc.item()})

        torch.save(self.model.state_dict(), os.path.join(self.parameters.log_path,self.parameters.model_name))
        self.logger.draw_graph()
        self.logger.writer.flush()
        self.logger.save_parameters_and_result()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
