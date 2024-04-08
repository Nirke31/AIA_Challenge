# NOTE!!!
I only kept this directory as reference for my own submission. Additionally, I am using the [**evaluation.py**](./evaluation.py)
file in my scripts. The only change (to my knowledge) was to the evaluation.py file. 


# Baseline submissions
Participants will be submitting their trained algorithm with their  code/workflow to generation predictions into the competition platform, in the programming language they prefer. Here we provide Docker containers and example submissions for common Python and Matlab, but participants are allowed to submit with other custom containers too. 

## Data visualization notebook
The `data_visualization.ipynb` guides the challengers in loading, manipulating, and visualizing the training data and labels.

## Code for baseline submission using a Machine learning model
See the folder `ml_python` to reproduce a working example of a ML-based submission written in Python.

## Code for baseline submission using a Heuristic-based approach
See the folder `heuristic_python` to reproduce a working example of a heuristic-based baseline written in Python. Note: no submission example are provided for the heuristic-based approach.

## Evaluation script
The `evaluation.py` script provides a standard way to assess the performance of the models submitted for the challenge. It employs metrics and evaluation techniques that are aligned with the challenge's objectives.

The toy datasets `ground_truth_toy.csv` and `participant_toy.csv` serve as simplified, 
example datasets for the challenge. These datasets are intended for initial 
testing and understanding of the evaluation script and the baseline model.

### Example Usage
The `run_evaluator` function is the main entry point of the script and accepts the following arguments:

- `participant`: Path to the participant's CSV file.
- `ground_truth`: Path to the ground truth CSV file.
- `plot_object`: Object ID for which to plot evaluation details.

You can also run the script directly from the command line. For example:
```bash
python evaluation.py --participant=participant.csv --ground_truth=ground_truth.csv --plot_object=12345
```
This example assumes you have a `participant.csv` and `ground_truth.csv` in the expected directories, and you want to plot evaluation details for object ID `12345`. If no arguments are provided, the evaluation will be run for the toy datasets.

### Returns
The `score` function within the file returns the evaluation metrics as per the challenge guidelines (that is, the F2 and the RMSE). Additionally, the precision and recall are also returned, and, if the `plot_object` parameter is provided, it generates plots for that specific object ID to aid in understanding the evaluation.
