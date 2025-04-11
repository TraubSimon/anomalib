# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

import sys 
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/src/")
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/")


# 1. Import required modules
from pathlib import Path

from src.anomalib.data import SensoPartAD
from src.anomalib.engine import Engine
from src.anomalib.models import EfficientAd

# 2. Initialize the model and load weights
model = EfficientAd()
engine = Engine()

# 3. Prepare test data
## MVTec Datamodule
datamodule = SensoPartAD(
    root="/home/development/Documents/Data/AnomalyDatasets/SensoPartAD/",
    category="2_Anwesenheit_und_Vollstaendigkeit",
    sub_category="2_5_Zahoransky_Completeness_MP",
    train_batch_size=1,  # as efficient ad only takes one image per batch
    num_workers=8,
)


# 4. Get predictions
test_result = engine.test(
    model=model,
    datamodule=datamodule,
    ckpt_path="/home/development/Documents/AnomalyDetection/anomalib/results/EfficientAd/SensoPartAD/2_Anwesenheit_und_Vollstaendigkeit/latest/weights/lightning/model.ckpt",
)

print(test_result)
