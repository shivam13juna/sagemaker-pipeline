from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json
import traceback

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime


class IntelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_dict, transform=None):

        self.root_dir = Path(root_dir)

        self.data_files = list((self.root_dir).glob("*/*"))
        # print("The data files were: ", self.data_files)

        self.transform = transform
        self.label_dict = label_dict
        self.classes = list(label_dict.keys())

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):

        # Load the image and label at the given index
        image = np.array(Image.open(self.data_files[index]))
        # convert to numpy array

        label = self.label_dict[self.data_files[index].parent.stem]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
#     def __getitem__(self, index):
#         try:
#             image = np.array(Image.open(self.data_files[index]))
#             # convert to numpy array

#             label = self.label_dict[self.data_files[index].parent.stem]

#             if self.transform:
#                 # print("going transform:", self.transform)
#                 transformed = self.transform(image=image)
#                 # print("going transform phase 2")
#                 image = transformed["image"]
#                 # print("going transform phase 3")

#         except Exception as e:
#             print("This is the error: ", e)
#             # print("This is the index: ", index)
#             # print("This is the data_files index: ", self.data_files[index])
#             # print("This is the label: ", label)
#             # print("This is the image: ", image)
#             # print traceback complete error message
#             print(traceback.format_exc())
#             # print only error message
#             print(sys.exc_info()[0])
#             raise e

class IntelCapstoneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str = "data/",
        test_data_dir: str = "data/",
        albumentations=None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.albumentations = albumentations
        if self.albumentations is None:
            self.albumentations = "[]"
        self.label_dict = {
                "buildings": 0,
                "forest": 1,
                "glacier": 2,
                "mountain": 3,
                "sea": 4,
                "street": 5,
            }
        

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        albumentations = json.loads(albumentations.replace("'", '"'))
        
        transform_train = []
        if len(albumentations) >= 1:
            for method in albumentations:
                if type(method) == dict:
                    transform_train.append(getattr(A, method.pop('name'))(**method))
                else:
                    transform_train.append(getattr(A, method)())
                    
        transform_train.append(A.Resize(128, 128))
        transform_train.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_train.append(ToTensorV2())

        self.transform_train = A.Compose(transform_train)
        

        self.transform_test = A.Compose(
            [   A.Resize(128, 128),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.data_train.classes)

    @property
    def classes(self):
        return self.data_train.classes

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            trainset = IntelDataset(
                self.train_data_dir,
                label_dict=self.label_dict,
                transform=self.transform_train,
            )
            testset = IntelDataset(
                self.test_data_dir,
                label_dict=self.label_dict,
                transform=self.transform_test,
            )

            self.data_train, self.data_test = trainset, testset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
