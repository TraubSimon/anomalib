# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Training using the Python API.

This example shows the basic steps to train an anomaly detection model
using the Anomalib Python API.
"""

# 1. Import required modules
import sys  
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/src/")
sys.path.append("/home/development/Documents/AnomalyDetection/anomalib/")

import csv
import gc
import logging 
import datetime
import time
import torch

logger = logging.getLogger(__name__)

# 1. loading the datamodules
from src.anomalib.data import SensoPartAD
from src.anomalib.models import Patchcore
from src.anomalib.engine import Engine

CATEGORIES = (
    "2_Anwesenheit_und_Vollstaendigkeit",
    "3_Montagepruefung", 
    "4_Leerkontrolle",
    "5_Qualitaet_allgemein",
    "6_Quality_Oberflaeche",
    "7_Bauch_Ruecken",
)

SUB_CATEGORIES = (
    [
        '2_1_BMW_Blechmontage', 
        '2_2_Presence_O-Ring', 
        '2_3_BMW-USA_Schweissnaht', 
        '2_4_SP-USA_WeldSeam_presence', 
        '2_5_Zahoransky_Completeness_MP', 
        '2_6_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_li', 
        # '2_7_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_re', 
        '2_8_Weckerle', 
    ],
    [    
        '3_1_Tritecnica_Colored_pen', 
        #'3_2_DAIMLER_Pyro', 
        '3_3_WeilEN_Rohre_', 
        '3_4_KrausMaffeySchweden_ARu', 
        '3_5_DAIMLER_Steuereinheit_Positionierung', 
        #3_6_BMW-USA_Montagepruefung',
        '3_7_Tuben', 
    ],
    [
        '4_1_CJa_Hengst',
    ], 
    [
        '5_1_SP_FR_Print1', 
        '5_1_SP_FR_Print2', 
        '5_2_Tritecnica_Electric_contacts', 
        # '5_3_SP_FR_42', 
        '5_4_Tritecnica_aerosol', 
        '5_5_Tritecnica_cable_cam35', 
        '5_6_BMW_Gummidichtung_LoecherTHD_G02_12-04-23', 
        '5_7_ARu AUDI Griffe', 
        # '5_8_DAIMLER_Pyrosicherung'
    ], 
    [
        '6_1_Kunstleder_Diss_KBE', 
        '6_2_BMW_MetalNut', 
        '6_3_curaprox_ARu_Druck_close', 
        '6_4_curaprox_ARu_Druck_c', 
        '6_5_Hella_PlasticBurn', 
        #'6_6_Steatit-Bruch', 
        '6_7_SPKorea_Welding',
    ], 
    [
        '7_1_SP_FR_41', 
        '7_2_SensoPart_Zufuehrung_Kunststoffteil',
    ]
)


# save results in csv file
time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
with open(f'development/results/patchCore_SensoPartAD{time_now}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["subcategory", "image_AUROC", "image_F1Score", "training_time"])
    
    ## go through all categoreis and subcategories
    for sub_cats, cat in zip(SUB_CATEGORIES, CATEGORIES):
        for sub_cat in sub_cats:
            logger.info(f"Using category {cat}/{sub_cat}")
            gc.collect()
            
            
            # 1.Load SensoPart AD datamodule
            sensopart_datamodule = SensoPartAD(
                root="/home/development/Documents/Data/AnomalyDatasets/SensoPartAD/",
                category=cat,
                sub_category=sub_cat,
                train_batch_size=1,
                num_workers=8,
                num_train_imgs=32,
            )
            # 2. Initialize the model
            # EfficientAd is a good default choice for beginners
            model = Patchcore(
                pre_trained=False,
            )                                  

            # 3. Create the training engine
            engine = Engine()  # Train for 10 epochs

            # 4. Train the model
            start_time = time.time()
            engine.fit(datamodule=sensopart_datamodule, model=model)
            end_time = time.time()
            duration = end_time - start_time

            # 6. Test the model
            test_result = engine.test(
                model=model,
                datamodule=sensopart_datamodule,
                ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
            )
            
            torch.cuda.empty_cache()

            
            # log the test results:
            writer.writerow([
                sub_cat, 
                test_result[0]["image_AUROC"], 
                test_result[0]["image_F1Score"],
                duration]),
                
    
