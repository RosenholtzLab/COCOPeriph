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
parser = argparse.ArgumentParser(description="trainscratch settings")
parser.add_argument(
       "--train-ecc",
       type=int,
       default=5,
       help="eccentricity to train on",
   )
parser.add_argument(
       "--num-workers",
       type=int,
       default=4,
       help="numnber of workers for the dataloader",
   )
parser.add_argument(
       "--config-path",
       type=str,
       default='COCO-Detection/faster_rcnn_R_50_FPN_1x_scratch_shuffle_ecc100.yaml',
       help="path to config yaml",
   )
parser.add_argument(
       "--model-weights-file",
       type=str,
       default=f"output/trainscratch_ecc100/model_final.pth",
       help="name of weights file to use",
   )
args = parser.parse_args()
train_ecc = args.train_ecc
if train_ecc == -1:
    train_ecc = "baseline"
# train_ecc = 10

print("testing scratchtrained model")
test_eccs = [0,5,10,15,20]



#### Load in training configuration
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #could change this later to somethign more state of the art!
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = args.model_weights_file.replace('/model_final.pth','')
print("output dir:", cfg.OUTPUT_DIR)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#cfg.MODEL.WEIGHTS = '/home/gridsan/vdutell/.torch/iopath_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl' #baseline model
#model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = '/home/gridsan/groups/RosenholtzLab/detection_repos/detectron2/model_weights/R_50_FPN_3x/model_final_280758.pkl'
#load finetuned model
cfg.MODEL.WEIGHTS = args.model_weights_file # path to the model we just trained
cfg.DATALOADER.NUM_WORKERS = args.num_workers
predictor = DefaultPredictor(cfg)
#outputs = predictor(test_im)
# cfg.SOLVER.BASE_LR = 0.0001 #reduce learning rate for fine-tuning


####RUN INFERENCE ON fine-tuned model
from detectron2.data import build_detection_test_loader
import pickle

# load finetuned model

base_path = '.'
val_results_list = []

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
    val_results_path = cfg.OUTPUT_DIR + f'/coco_val_2017_ecc{ecc}_eval/val_results.pkl'
    print(val_results_path)
    if(os.path.exists(val_results_path)):
        with open(val_results_path, 'rb') as f:
            val_results = pickle.load(f)
    else:
        os.makedirs(cfg.OUTPUT_DIR + f"/coco_val_2017_ecc{ecc}_eval", exist_ok=True)
        evaluator = COCOEvaluator(f"coco_val_2017_ecc{ecc}", output_dir=cfg.OUTPUT_DIR + f"/coco_val_2017_ecc{ecc}_eval")
        registeredcodatasetforevaluating = f"coco_val_2017_ecc{ecc}"
        val_loader = build_detection_test_loader(cfg, registeredcodatasetforevaluating)
        val_results = inference_on_dataset(predictor.model, val_loader, evaluator, registeredcodatasetforevaluating)
        #val_loader = build_detection_test_loader(cfg, f"coco_val_2017_ecc{ecc}")
        #val_results = inference_on_dataset(predictor.model, val_loader, evaluator)
        with open(val_results_path, 'wb') as f:
            pickle.dump(val_results, f)
    val_results_list.append(val_results)
