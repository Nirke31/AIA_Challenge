
<div align="center">

# AI SSA Challenge Submission - MiseryModel 

</div>

<div align="left">

This repo was cloned from the [splid-devkit repository](https://github.com/ARCLab-MIT/splid-devkit). It is the repository used in the submission of the 
'MiseryModel' in the AI SSA Challenge.

# Content
- [**own_model**](./own_model): This directory contains all relevant submission files. See directory for more information.
- [**dataset**](./dataset): Contains datasets. Currently uploaded are phase 1 version 2 and 3, as well as the phase 2
dataset.
- [**baseline_submissions**](./baseline_submissions): Only the NodeDetectionEvaluator class is used from evaluation.py
- [**environment.yml**](./environment.yml): This is the anaconda environment file extracted from the environment used 
for this submission.
- [**requirements.txt**](./requirements.txt): This file is the pip version of the environment.yml.
file.

# Reproducibility
#### [WIP]
Currently, the only way to train all models and run the evaluation, one has to clone/download the whole repo, go into
the [own_file/src/](./own_model/src) directory and execute the respective python file. See [own_file/src/](./own_model/src) 
for more information.

Sadly, as of submitting the respective Technical Report to this repository, I was not able to finish an easier training 
and evaluation. The current status is on the branch "easyRun". The training can already be started by executing a single
file, however, in the process of restructuring, the training of the CNN models broke and the scoring is significantly lower.
The bug could not be found by now.



</div>


