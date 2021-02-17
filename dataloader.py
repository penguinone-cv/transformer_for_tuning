import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


class DataLoader():
    def __init__(self, data_path, batch_size, img_size, num_workers, pin_memory):
        transform = transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, ), (0.5, ))])
        train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        val_data = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        self.dataloaders = {"train": train_loader, "val": val_loader}
