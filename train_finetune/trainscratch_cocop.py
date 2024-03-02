## load requirements
import sys, os, distutils.core
import torch, detectron2
import cv2
import matplotlib.pyplot as plt

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import argparse

##What are we doing?
parser = argparse.ArgumentParser(description="finetuning settings")
parser.add_argument(
        "--train-ecc",
        type=int,
        default=100,
        help="eccentricity to fine tune on",
    )
parser.add_argument(
        "--config-file",
        type=str,
        default="COCO-Detection/faster_rcnn_R_50_FPN_1x_scratch_shuffle_ecc100.yaml",
        help="configuration file path(yaml)",
    )
args = parser.parse_args()

train_ecc = args.train_ecc

print("finetuning on ",train_ecc)
test_eccs = [0,5,10,15,20]



#### Load in training configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
cfg.OUTPUT_DIR = f'./output/trainscratch_ecc{train_ecc}'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# now the weights are in custom config files

base_path = '.'

# ###### Train on our desired eccentricity

register_coco_instances(f"coco_train_2017_ecc0", {}, f'{base_path}/detrex/coco/annotations/instances_train2017.json',
                       f'{base_path}/detrex/coco/train2017')

register_coco_instances(f"coco_train_2017_ecc5", {}, f'{base_path}/detrex/coco/annotations/instances_train2017.json',
                        f'{base_path}/detrex/coco/train_2017_5')

register_coco_instances(f"coco_train_2017_ecc10", {}, f'{base_path}/detrex/coco/annotations/instances_train2017.json',
                        f'{base_path}/detrex/coco/train_2017_10')

register_coco_instances(f"coco_train_2017_ecc15", {}, f'{base_path}/detrex/coco/annotations/instances_train2017.json',
                        f'{base_path}/detrex/coco/train_2017_15')

register_coco_instances(f"coco_train_2017_ecc20", {}, f'{base_path}/detrex/coco/annotations/instances_train2017.json',
                        f'{base_path}/detrex/coco/train_2017_20')


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg);
trainer.resume_or_load(resume=False);
trainer.train()

print('All Done Training!')
