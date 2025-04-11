# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Training using the Python API.

This example shows the basic steps to train an anomaly detection model
using the Anomalib Python API.
"""

# 1. Import required modules
import csv
import sys 
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/src/")
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/")

import logging 
import datetime
logger = logging.getLogger(__name__)

# 1. loading the datamodules
from src.anomalib.data import MVTec
from src.anomalib.models import EfficientAd
from src.anomalib.engine import Engine

time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
with open(f'development/results/efficientAD_MVTec{time_now}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["category", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"])
    CATEGORIES = (
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    )   

    for category in CATEGORIES:    
        ## MVTec Datamodule
        mvtec_datamodule = MVTec(
            root="/home/development/Documents/Data/AnomalyDatasets/MVTec/",
            category=category, 
            train_batch_size=1,  # as efficient ad only takes one image per batch
            num_workers=8,
        )

        # 3. Initialize the model
        # EfficientAd is a good default choice for beginners
        model = EfficientAd(imagenet_dir='/home/development/Documents/Data/AnomalyDatasets/imagenette')                                  

            
        # 4. Create the training engine
        engine = Engine(max_epochs=10) 

        # 5. Train the model
        engine.fit(datamodule=mvtec_datamodule, model=model)

        # 6. Test the model
        test_result = engine.test(
            model=model, 
            datamodule=mvtec_datamodule,
            ckpt_path="/home/development/Documents/AnomalyDetection/anomalib/results/EfficientAd/MVTec/bottle/latest/weights/lightning/model.ckpt",
        )
        
        # log the results
        writer.writerow([
            category,
            test_result[0]["image_AUROC"], 
            test_result[0]["image_F1Score"], 
            test_result[0]["pixel_AUROC"], 
            test_result[0]["pixel_F1Score"]])
        