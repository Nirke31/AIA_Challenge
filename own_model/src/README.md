# Structure

This directory is quite convoluted and many files are not actively in use. This is due to lots of testing and 
restructuring the submission model.

The three main files are the following:
1. [**submission.py**](./submission.py): This file is executed for every prediction. It is the tidiest and most easy to understand file, which
also follows the structure described in the associated paper to this submission.
Currently, one should be able to just run it and get prediction results on the full model. Else, to use other models, 
the file paths of the models in line 376, 377, 383, 385, 387, 389 and 391 have to be change to the model one wants to use.
For example, to use the submission model for the EW changepoint prediction, change "state_classifier_EW_full.joblib" to
"state_classifier_EW_xgboost.joblib". The same goes for all other models. 
The data path for the test dataset can be changed in line 23. For example, to run the submission test, just change '/test_own/'
to '/test/'.
The 'TEST_LABEL_PATH' should point to the test labels and is used for evaluation in the NodeDetectionEvaluator. To run 
the submission evaluation this path should also be adapted to the 'test_label'.
2. [**findChange.py**](./findChange.py): This file is used for training the changepoint predicition models (both EW and NS directions).
For training, one only has to set the global variables accordingly. 
   * _TRAIN_DATA_PATH_ and _TRAIN_LABEL_PATH_ have to point
   to the train dataset and label set. 
   * _DIRECTION_ ("EW" or "NS") describes for which direction the model is currently trained. 

    The output model is just dumped via joblib.dump into "../trained_model" with the name "state_classifier.joblib". One has 
   to rename its model if multiple models are trained.
3. [**train_classifier.py**](./train_classifier.py): This file is used for training the behaviour classification models, both EW and NS direction 
for both changepoints and first samples. Similar to _findChange.py_, the training parameters are set via global variables. 
   * _TRAIN_DATA_PATH_ and _TRAIN_LABEL_PATH_ have to point
   to the train dataset and label set. 
   * _DIRECTION_ ("EW" or "NS") describes for which direction the model is currently trained. 
   * _TRAINED_MODEL_PATH_ points to the stored classification model with name _TRAINED_MODEL_NAME_. (Note: often used
checkpoints instead of stored model due to overfitting, see below)
   * _FIRST_, if set to true, classifiers for beginning samples are trained, if set to false, changepoint models are trained.

    Two things should be noted. First, the input data is scaled and the scaler is dumped again via joblib.dump into
"../trained_model" with the name "scaler.joblib". This scaler is used for scaling the test data. Second, the model is
trained with early stopping and model checkpoint of pytorch lightning. Due to a "late" early stopping, I often used the 
model checkpoint instead of the model that is actually stored at the end of the run. The reason is to avoid overfitting
on the train set. 

All other files are quickly listed below in (more or less) descending importance:
* [**dataset_manip.py**](./dataset_manip.py): All initial train dataset manipulation are done in this file. Additionally, contains the 
specific datasets for training all CNNs.
* [**myModel.py**](./myModel.py): Contains CNN model used in training. Also contains older models, currently not used such as an 
Autoencoder, Transformer and simpler changepoint classifier.
* [**optuna_hsearch.py**](./optuna_hsearch.py): The [optuna library](https://optuna.org/) was used for finding the best hyperparameters for 
the changepoint classifier. 
* [**test.py**](./test.py): File where many small tests were carried out.
* [**multiScale1DResNet.py**](./multiScale1DResNet.py): Initially, a 1D ResNet was tested but model was deemed to complex. Now houses the "DumbNet",
which is just a simple fully connected neural network.
* [**featureEncoding.py**](./featureEncoding.py): [DEPRECATED] - Trained the Autoencoder with this file. Not successful but with further work
could prove beneficial.
* [**first_sample.py**](./first_sample.py): [DEPRECATED] - Tried to classify the behaviour of first samples via RandomForest. Not successful.
* [**findChangeViaConv.py**](./findChangeViaConv.py): [DEPRECATED] - Tried to find changepoints via CNN. Not successful. 
* [**transformer.py**](./transformer.py): [DEPRECATED] - Initial try of the challenge. Transformer was deemed to complex and not suited for 
dataset. Not successful. 
