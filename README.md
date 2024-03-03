# COCOPeriph
Official Repository for COCO-Periph: Bridging the Gap Between Human and Machine Perception in the Periphery ICLR 2024

Citation:  
Harrington, Anne, et al. "COCO-Periph: Bridging the Gap Between Human and Machine Perception in the Periphery." The Twelfth International Conference on Learning Representations. 2023.

Paper Here:  
[https://openreview.net/forum?id=MiRPBbQNHv&noteId=hPqieNU8QA](https://openreview.net/forum?id=MiRPBbQNHv&noteId=hPqieNU8QA)

Human Psychophysics Experiment Repo (Matlab PsychToolBox) Here:
[https://github.com/RosenholtzLab/CocoPsychExp](https://github.com/RosenholtzLab/CocoPsychExp)

Uniform Texture Tiling Model (Matlab) Here:
[https://github.com/RosenholtzLab/TTM/tree/dataset_generation](https://github.com/RosenholtzLab/TTM/tree/dataset_generation)

Dataset, Model Weights, and Psychophysics Experiment Images Hosted Here:
[https://data.csail.mit.edu/coco_periph/](https://data.csail.mit.edu/coco_periph/) (put them in ./psychophysics_experiment/stimuli), (./model_weights), and  (./psychophysics_experiment/human_results)

Codebase Atlas:
- **SewMongrelExperiment.ipynb**  Use this to create pseudofoveated images by 'sewing' image transforms at increasing eccentricities together. This notebook specifically creates the pseudofoveated images used in the human and machine psychophysics experiments. You can re-create these pseudofoveated images with this notebook OR you can download pre-generated ones (here)[https://data.csail.mit.edu/coco_periph/]
- **train_finetune/*.py** Trains, Finetunes, and evaluates models on COCO-Periph
- **Get AP Vals.ipynb** Read evalution results from pkl files for AP.
- **MachinePsychophysicsSignalDetectionTheory** Run models through machine psychophysics experiment for uniform COCO-Periph Images
- **MachinePsychophysicsSignalDetectionTheoryFoveated** Run models through machine psychophysics experiment for foveated images (setup to run classic TTM Foveated images, uncomment a few lines to run for Sewn Pseudofoveated images)
- **CombineHumanData.ipynb** Analyize raw human experiment data (download data from (here)[https://data.csail.mit.edu/coco_periph/])
- **CompareHumanMachinePsychophysics.ipynb** Compare data from machine psychophysics to human psychcophysics to creates figures from paper.
- **corruption/finetune_evalution(_corruption,_trainingset,_plot).py** Evaluates various models for corruption robustness
- **Detectron2CorruptionPlot.ipynb** Plots corruption results
- **DemoPseudofoveatedRings.ipynb** Creates Paper demo plot with rings overlaying image
- **MakeTeaserFigure.ipynb** Creates paper teaser figure with models demod on 

Model Nomenclature for Train/Finetuned RCNN Models:
-1: Baseline
-2: Trained from Scratch on all eccentricities images from COCO-Periph training set
0: Baseline Model Fine-tuned on 0 degree eccenricity (original) images from original MS-COCO training set (control condition for fine-tuning)
5:  Baseline Model Fine-tuned on 5 degree eccentricity images from COCO-Periph training set
10:  Baseline Model Fine-tuned on 10 degree eccentricity images from COCO-Periph training set
15:  Baseline Model Fine-tuned on 15 degree eccentricity images from COCO-Periph training set
20:  Baseline Model Fine-tuned on 20 degree eccentricity images from COCO-Periph training set
100: Baseline Model Fine-turned on all eccentricities images from COCO-Periph training set, largest subset of images available at time of submission.
101: Baseline Model Fine-turned on all eccentricities images from COCO-Periph training set, smaller subset of images with number of images between eccentricities all equal.



