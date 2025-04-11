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

from src.anomalib.data import PredictDataset
from src.anomalib.engine import Engine
from src.anomalib.models import EfficientAd

# 2. Initialize the model and load weights
model = EfficientAd()
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of images
dataset = PredictDataset(
    # path=Path("/home/development/Documents/Data/AnomalyDatasets/MVTec/bottle/test/good/"),
    path=Path("/home/development/Documents/Data/AnomalyDatasets/MVTec/bottle/test/broken_large/"),
    image_size=(256, 256),
)

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="/home/development/Documents/AnomalyDetection/anomalib/results/EfficientAd/MVTec/bottle/latest/weights/lightning/model.ckpt",
)

# 5. Access the results
if predictions is not None:
    for prediction in predictions:
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
        print(f"pred_label {pred_label} and pred_score: {pred_score}")