# Structure

This directory contains the trained changepoint and classification models. Currently, two variants can be found.

All models containing 'full' in their name are models that were trained on the phase 2 dataset with a random train test 
split of 0.8. 
The other models are the submission models. 

'scaler_full.joblib' is the scaler that is used to scale the classification models for the phase 2 dataset. The 
'scaler_submission.job' scaler, is the scaler that was used in the submission. 
