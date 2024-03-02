## 10 degrees
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
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

#set random seed for reproducibility and comparison between runs
torch.manual_seed(0)

#What are we doing?
parser = argparse.ArgumentParser(description="finetuning settings")
parser.add_argument(
       "--num-workers",
       type=int,
       default=4,
       help="numnber of workers for the dataloader",
   )
parser.add_argument(
       "--config-path",
       type=str,
       default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
       help="path to config yaml",
   )
parser.add_argument(
       "--model-weights-path",
       type=str,
       default="./model_final.pth",
       help="name of weights file path to use",
   )
parser.add_argument(
       "--model-name",
       type=str,
       default="FPN",
       help="name of weights file to use",
   )
args = parser.parse_args()
# train_ecc = args.train_ecc
# if train_ecc == -1:
#     train_ecc = "baseline"
# # train_ecc = 10

# print("finetuning on ",train_ecc)
test_eccs = [0,5,10,15,20]



#### Load in training configuration
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #could change this later to somethign more state of the art!
cfg.merge_from_file(model_zoo.get_config_file(args.config_path))
path_parts = args.model_weights_path.split('/')
path = ''
for p in path_parts[:-1]:
    path += p + '/'
path = path[:-1]
print(path)
cfg.OUTPUT_DIR = path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.DATALOADER.NUM_WORKERS = args.num_workers
cfg.MODEL.WEIGHTS = args.model_weights_path  # path to the model we just trained
cfg.SOLVER.BASE_LR = 0.0001 #reduce learning rate for fine-tuning
predictor = DefaultPredictor(cfg)
#outputs = predictor(test_im)



####RUN INFERENCE ON fine-tuned model
from detectron2.data import build_detection_test_loader
import pickle

val_results_list = []
base_path = '.'


#all the below in a loop
for ecc in test_eccs:
    
    ## Register Val Loaders
    if(ecc == 0):
        register_coco_instances(f"coco_val_2017_ecc{ecc}", {}, f'{base_path}/detrex/coco/annotations/instances_val2017.json',
                        f'{base_path}/detrex/coco/val2017')

    else:
        register_coco_instances(f"coco_val_2017_ecc{ecc}", {}, f'{base_path}/detrex/coco/annotations/instances_val2017.json',
                            f'{base_path}/detrex/coco/val2017_{ecc}')
    
    #Evaluate Performance on each eccentricity
    val_results_path = path + f'/coco_val_2017_ecc{ecc}_eval/val_results.pkl'
    if(os.path.exists(val_results_path)):
        with open(val_results_path, 'rb') as f:
            val_results = pickle.load(f)
    else:
        evaluator = COCOEvaluator(f"coco_val_2017_ecc{ecc}", output_dir=path + f"/coco_val_2017_ecc{ecc}_eval")
        registeredcodatasetforevaluating = f"coco_val_2017_ecc{ecc}"
        val_loader = build_detection_test_loader(cfg, registeredcodatasetforevaluating)
        val_results = inference_on_dataset(predictor.model, val_loader, evaluator, registeredcodatasetforevaluating)
        #val_loader = build_detection_test_loader(cfg, f"coco_val_2017_ecc{ecc}")
        #val_results = inference_on_dataset(predictor.model, val_loader, evaluator)
        with open(val_results_path, 'wb') as f:
            pickle.dump(val_results, f)
    val_results_list.append(val_results)
