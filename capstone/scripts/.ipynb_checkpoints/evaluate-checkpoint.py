from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json
import tarfile
import numpy as np

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A
from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

from model import LitResnet
from dataset import IntelCapstoneDataModule

ml_root = Path("/opt/ml")

model_artifacts = ml_root / "processing" / "model"
dataset_dir = ml_root / "processing" / "test"

# def eval_model(trainer, model, datamodule):
#     test_res = trainer.test(model, datamodule)[0]
    
#     report_dict = {
#         "multiclass_classification_metrics": {
#             "accuracy": {
#                 "value": test_res["test/acc"],
#                 "standard_deviation": "0",
#             },
#         },
#     }
    
#     eval_folder = ml_root / "processing" / "evaluation"
#     eval_folder.mkdir(parents=True, exist_ok=True)
    
#     out_path = eval_folder / "evaluation.json"
    
#     print(f":: Writing to {out_path.absolute()}")
    
#     with out_path.open("w") as f:
#         f.write(json.dumps(report_dict))

def eval_model(trainer, model, datamodule):
    test_res = trainer.test(model, datamodule)[0]

    # Define the transformations
    transform = A.Compose(
        [
            A.GaussNoise(mean=[0.485, 0.456, 0.406], p=0.5),
            A.RandomBrightness(limit=0.7, p=0.5),
        ]
    )

    # Apply the transformations to the test set
    test_data = np.stack([datamodule.data_test[i][0] for i in range(len(datamodule.data_test))])
    test_label = np.stack([datamodule.data_test[i][1] for i in range(len(datamodule.data_test))])

    # Apply the transformations to the test set

    mod_test_data = []
    for i in range(len(test_data)):
        # Default image format (both input and output) for albumentations is (H, W, C), but the image shape in test_data is (C, H, W)
        value = transform(image=test_data[i].transpose(1, 2, 0))['image'] # Converting image from (C, H, W) to (H, W, C)
        value = value.transpose(2, 1, 0) # Converting image from (H, W, C) to (C, H, W)
        mod_test_data.append(value)

    mod_test_data = np.stack(mod_test_data)
    # print(mod_test_data.shape)
    
    preds = model.forward(torch.from_numpy(mod_test_data[:1000])) 

    acc  = (np.argmax(preds.detach().numpy(), axis=1) == test_label[:1000]).sum() / len(test_label[:1000])

    acc = round(acc, 3)


    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": test_res["test/acc"],
                "standard_deviation": "0",
            },
            "robustness": {
                "value": acc,
                "standard_deviation": "0",
                }
        },
    }
    
    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    out_path = eval_folder / "evaluation.json"
    
    print(f":: Writing to {out_path.absolute()}")
    
    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))
        
        
    
#     eval_folder = Path("tmp")
    
#     out_path = eval_folder / "evaluation.json"
    
#     print(f":: Writing to {out_path.absolute()}")
    
#     with out_path.open("w+") as f:
#         f.write(json.dumps(report_dict))
    


   

if __name__ == '__main__':
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    # datamodule = IntelCapstoneDataModule(
    #     train_data_dir=train_channel,
    #     test_data_dir=test_channel,
    #     batch_size=int(hyperparameters.get("batch_size")),
    #     num_workers=1,
    #     albumentations=hyperparameters.get("augmentations").replace("'", '"')
    # )
    datamodule = IntelCapstoneDataModule(
        train_data_dir=dataset_dir.absolute(),
        test_data_dir=dataset_dir.absolute(),
        num_workers=os.cpu_count()
    )
    datamodule.setup()
    
    model = LitResnet.load_from_checkpoint(checkpoint_path="last.ckpt")
    
    trainer = pl.Trainer(
        accelerator="auto",
    )
    
    print(":: Evaluating Model")
    eval_model(trainer, model, datamodule)