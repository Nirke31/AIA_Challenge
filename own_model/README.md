This directory is split into two subdirectories:
* [**src**](./src) contains all source files that were created for the submission of the MiseryModel
* [**trained_model**](./trained_model) contains trained models which are used in the evaluation of test data. It is also 
the direction in which trained models are stored. Additionally, it houses the test data scaler.  

The present dockerfile was used to build a docker image which was pushed to the [evalai](https://eval.ai/web/challenges/challenge-page/2164/overview) 
website.

The [**requirements.txt**](./requirements.txt) file contains all necessary libaries for running evaluations (which is 
the ./src/submission.py file). It was also used in the docker build. 