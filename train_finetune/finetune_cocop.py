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
        default=5,
        help="eccentricity to fine tune on",
    )
parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="starting learning rate",
    )
parser.add_argument(
        "--config-file",
        type=str,
        default="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
        help="configuration file path(yaml)",
    )
args = parser.parse_args()

train_ecc = args.train_ecc

print("finetuning on ",train_ecc)
test_eccs = [0,5,10,15,20]



#### Load in training configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
cfg.OUTPUT_DIR = f'./output/finetune_ecc{train_ecc}'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# now the weights are in custom config files
# cfg.MODEL.WEIGHTS = '/home/gridsan/groups/RosenholtzLab/detection_repos/detectron2/model_weights/R_50_FPN_3x/model_final_280758.pkl'
#outputs = predictor(test_im)
cfg.SOLVER.BASE_LR = args.lr #reduce learning rate for fine-tuning
print("Learning rate:", args.lr)

# ###### Train on our desired eccentricity

coco_annotations_folder = '.'

register_coco_instances(f"coco_train_2017_ecc0", {}, f'{coco_annotations_folder}/detrex/coco/annotations/instances_train2017.json',
                        f'{coco_annotations_folder}/detrex/coco/train2017')

register_coco_instances(f"coco_train_2017_ecc5", {}, f'{coco_annotations_folder}/detrex/coco/annotations/instances_train2017.json',
                        f'{coco_annotations_folder}/detrex/coco/train_2017_5')

register_coco_instances(f"coco_train_2017_ecc10", {}, f'{coco_annotations_folder}/detrex/coco/annotations/instances_train2017.json',
                        f'{coco_annotations_folder}/detrex/coco/train_2017_10')

register_coco_instances(f"coco_train_2017_ecc15", {}, f'{coco_annotations_folder}/detrex/coco/annotations/instances_train2017.json',
                        f'{coco_annotations_folder}/detrex/coco/train_2017_15')

register_coco_instances(f"coco_train_2017_ecc20", {}, f'{coco_annotations_folder}/detrex/coco/annotations/instances_train2017.json',
                        f'{coco_annotations_folder}/detrex/coco/train_2017_20')
 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg);
trainer.resume_or_load(resume=False);
trainer.train()


# # # Plot Results (after running evalution script

# eccs = [0,5,10,15,20]
# aps = [v['bbox']['AP'] for v in val_results_list]
# aps_l = [v['bbox']['APl'] for v in val_results_list]
# aps_m = [v['bbox']['APm'] for v in val_results_list]
# aps_s = [v['bbox']['APs'] for v in val_results_list]
# aps_50 = [v['bbox']['AP50'] for v in val_results_list]
# aps_75 = [v['bbox']['AP75'] for v in val_results_list]

# plt.plot(eccs, aps, c='black',label='AP All')
# plt.plot(eccs, aps_50, label='AP 50')
# plt.plot(eccs, aps_75, label='AP 75')
# plt.plot(eccs, aps_s, label='AP Small')
# plt.plot(eccs, aps_m, label='AP Medium')
# plt.plot(eccs, aps_l, label='AP Large')
# plt.xlabel('Eccentricity')
# plt.ylabel('Average Precision')
# plt.title('Detectron2 5deg FineTuned Model Object Detection Performance') 
# plt.legend()
# plt.tight_layout()

# plt.savefig('./AP_ecc5train_plot.png')
