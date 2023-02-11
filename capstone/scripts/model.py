from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json
import numpy as np

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

class LitResnet(pl.LightningModule):
    def __init__(self, model_name='resnet18', optimizer = 'adam', num_classes=10, lr=0.05):
        super().__init__()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model_name = model_name
        self.optimizer = optimizer
        self.model = timm.create_model(model_name=self.model_name, pretrained=True, num_classes=num_classes)
        self.lr = lr
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Resize((128, 128)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.to_tensor = T.ToTensor()
        # self.resize = T.Resize((128, 128))
        # self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = A.Compose(
            [   A.Resize(128, 128),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        

    def forward(self, x):
        # print("The shape of x is: ", x.shape, type(x))
        # x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        # transformed = []

        # if x.ndim == 4:
        #     # print("The shape of x is 4", x.shape, x.ndim)
        #     for i in range(x.shape[0]):
        #         # print("The shape of x pre resize is", x[i].shape)
        #         # x[i] = A.Resize(128, 128)(image=x[i])['image']
        #         # print("The shape of x after resize is", x[i].shape)
        #         # x[i] = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image=x[i])['image']
        #         # print("The shape of x after normalize is", x[i].shape)
        #         # x[i] = ToTensorV2()(image=x[i])['image']
        #         # print("The shape of x after ToTensorV2 is", x[i].shape)
        #         # x[i] = self.transform(image=x[i])['image']
        #         transformed.append(self.transform(image=x[i])['image'])
        # else:
        #     # print("The shape of x is not 4", x.shape, x.ndim)
        #     x = self.transform(image=x)['image']

        # inp = torch.from_numpy(np.stack(transformed)).to(self.device)
        # # print("The shape of inp is: ", inp.shape, type(inp))
        out = self.model(x)
        return F.log_softmax(out, dim=1)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):

        
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        return {"optimizer": optimizer}