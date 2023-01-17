from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

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

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model_name = model_name
        self.optimizer = optimizer
        self.model = timm.create_model(model_name=self.model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
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
        logits = self(x)
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