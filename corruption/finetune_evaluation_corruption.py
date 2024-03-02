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

from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from imagecorruptions import corrupt, get_corruption_names
import detectron2.data.transforms as T
from fvcore.transforms.transform import Transform
import pickle

class CorruptImage(Transform):
    def __init__(self, corruption_name, severity):
        """
        Args:

        """
        super().__init__()
        self._set_attributes(locals())
    def apply_image(self, image):
        return corrupt(image, corruption_name=self.corruption_name,severity=self.severity)
    def apply_coords(self, coords):
        return coords


class CorruptionAugmentation(T.Augmentation):
    def __init__(self, corruption_name, severity):
        """
        Args:

        """
        super().__init__()
        self.corruption_name = corruption_name
        self.severity= severity
    def get_transform(self, image):
        return CorruptImage(self.corruption_name, self.severity)

#What are we doing?
parser = argparse.ArgumentParser(description="finetuning settings")
parser.add_argument(
       "--train-ecc",
       type=int,
       default=5,
       help="eccentricity to fine tune on",
   )
parser.add_argument(
       "--num-workers",
       type=int,
       default=8,
       help="numnber of workers for the dataloader",
   )
parser.add_argument(
       "--config-path",
       type=str,
       default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
       help="path to config yaml",
   )
parser.add_argument(
       "--model-weights-file",
       type=str,
       default="model_final.pth",
       help="name of weights file to use",
   )
parser.add_argument(
       "--output-dir",
       type=str,
       default=None,
       help="path",
   )
args = parser.parse_args()
train_ecc = args.train_ecc

#### Load in training configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
if args.output_dir is None:
    cfg.OUTPUT_DIR = f'./output/finetune_ecc{train_ecc}'
else:
    cfg.OUTPUT_DIR = args.output_dir
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#load finetuned model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weights_file)  # path to the model we just trained
cfg.DATALOADER.NUM_WORKERS = args.num_workers
predictor = DefaultPredictor(cfg)
if args.output_dir is None:
    cfg.SOLVER.BASE_LR = 0.0001 #reduce learning rate for fine-tuning

#model_path
model_path = './'

####RUN INFERENCE ON fine-tuned model
## Register Val Loaders
register_coco_instances(f"coco_val_2017_corrupt", {}, f'{model_path}/detrex/coco/annotations/instances_val2017.json',
                   f'{model_path}/detrex/coco/val2017')

for corruption in get_corruption_names():
    for severity_level in range(5):

        #Evaluate Performance on each eccentricity
        if args.output_dir is None:
            val_results_path = f'{model_path}/trainscratch_ecc{train_ecc}/coco_val_2017_corruption_{corruption}_severity_{severity_level+1}_eval/val_results.pkl'
        else:
            output_dir = os.path.join(args.output_dir,f'coco_val_2017_corruption_{corruption}_severity_{severity_level+1}_eval')
            val_results_path = os.path.join(output_dir,'val_results.pkl')
        print(val_results_path, output_dir)
        if(os.path.exists(val_results_path)):
            with open(val_results_path, 'rb') as f:
                val_results = pickle.load(f)
        else:
            if args.output_dir is None:
                evaluator = COCOEvaluator(f"coco_val_2017_corrupt", output_dir=f'{model_path}/finetune_ecc{train_ecc}/coco_val_2017_corruption_{corruption}_severity_{severity_level+1}_eval')
            else:
                evaluator = COCOEvaluator(f"coco_val_2017_corrupt", output_dir=output_dir)
            mapper=DatasetMapper(cfg, is_train=False, augmentations=[CorruptionAugmentation(corruption, severity_level+1)])    
            val_loader = build_detection_test_loader(cfg, "coco_val_2017_corrupt" ,mapper=mapper)
            # val_loader = build_detection_test_loader(cfg, f"coco_val_2017_corrupt")
            val_results = inference_on_dataset(predictor.model, val_loader, evaluator)
            if os.path.isfile(val_results_path):
                raise Exception("file exists! don't overwrite")
            with open(val_results_path, 'wb') as f:
                pickle.dump(val_results, f)
