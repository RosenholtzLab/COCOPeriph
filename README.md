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

Dataset and Psychophysics Experiment Images Hosted Here:
[https://data.csail.mit.edu/coco_periph/](https://data.csail.mit.edu/coco_periph/) (put them in ./psychophysics_experiment/stimuli) and in (./psychophysics_experiment/human_results)

List of Notebooks:
- SewMongrelExperiment.ipynb  Use this to create pseudofoveated images by 'sewing' image transforms at increasing eccentricities together. This notebook specifically creates the pseudofoveated images used in the human and machine psychophysics experiments. You can re-create these pseudofoveated images with this notebook OR you can download pre-generated ones (here)[https://data.csail.mit.edu/coco_periph/]
- CombineHumanData.ipynb Use this to analyize raw human experiment data (download data from (here)[https://data.csail.mit.edu/coco_periph/]
- CompareHumanMachinePsychophysics.ipynb Use this to compare data from machine psychophysics to human psychcophysics to creates figures from paper. Contact Authors for access to raw data
- 



