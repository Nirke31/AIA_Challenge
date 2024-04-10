import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from own_model.src.dataset_manip import load_data
from own_model.src.findChange import main

# Can be set for debug. Load only partial data set or -1 for all
NUM_CSV = 100


def load_train_and_test(train_data_path: Path, train_label_path: Path, test_data_path: Path, test_label_path: Path):
    # load train and test data
    train: pd.DataFrame = load_data(train_data_path, train_label_path, amount=NUM_CSV)
    test: pd.DataFrame = load_data(test_data_path, train_label_path, amount=NUM_CSV)
    test_labels: pd.DataFrame = pd.read_csv(test_label_path)

    return train, test, test_labels


def load_train_and_split_test(train_data_path: Path, train_label_path: Path, split_ration: float):
    # load train and split into train and test
    df: pd.DataFrame = load_data(train_data_path, train_label_path, amount=NUM_CSV)

    object_ids = df['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids, test_size=1 - split_ration, random_state=42, shuffle=True)

    train_set = df.loc[train_ids].copy()
    test_set = df.loc[test_ids].copy()
    test_labels: pd.DataFrame = pd.read_csv(train_label_path)
    test_labels = test_labels.loc[test_labels["ObjectID"].isin(test_ids), :]

    return train_set, test_set, test_labels


def run_train(train_data_path: Path, train_label_path: Path, test_data_path: Path = None, test_label_path: Path = None,
              split: float = None):
    # load dataset(s)
    if split:
        df_train, df_test, df_test_labels = load_train_and_split_test(train_data_path, train_label_path, split)
    else:
        df_train, df_test, df_test_labels = load_train_and_test(train_data_path, train_label_path, test_data_path,
                                                                test_label_path)

    print("TRAIN CHANGEPOINT PREDICTION - DIRECTION EW")
    main(df_train, df_test, df_test_labels, "EW")

    print("TRAIN CHANGEPOINT PREDICTION - DIRECTION NS")
    main(df_train, df_test, df_test_labels, "NS")


# trains all models and stores them in own_model/trained_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train models")
    parser.add_argument("train_data_path", type=str,
                        help="Path to training data directory with .csv files. Please just supply an absolute path.")
    parser.add_argument("train_label_path", type=str,
                        help="Path to training label .csv file. Please just supply an absolute path.")
    parser.add_argument("test_data_path", type=str, nargs='?', default=None,
                        help="Path to test data directory with .csv files. Please just supply an absolute path.")
    parser.add_argument("test_label_path", type=str, nargs='?', default=None,
                        help="Path to training label .csv file. Please just supply an absolute path.")
    parser.add_argument("-s", "--split", type=float, default=None,
                        help="Splits training data into train test sets instead of using test_data_path for test data. "
                             "Input value must be a float between 0 and 1.")
    args = parser.parse_args()

    wrong_input = False
    # if split is set we only need train data and label
    if args.split:  # split is set
        # check that split is in range of 0.0 to 1.0
        if not 0.0 < args.split < 1.0:
            print("ERROR: Split ration must be in range of (0.0, 1.0).")
            wrong_input = True

    else:  # split is not set
        # check if we have all four data paths
        if args.test_data_path is None or args.test_label_path is None:  # at least one is None.
            print("ERROR: Please provide a path to test data and test labels as well. "
                  "Else set the -s flag to use part of train data as test set and only provide train data and labels.")
            wrong_input = True

    if wrong_input:
        wrong_input = False
        exit()

    a = Path(args.train_data_path)
    b = Path(args.train_label_path)
    c = Path(args.test_data_path) if args.test_data_path is not None else None
    d = Path(args.test_label_path) if args.test_data_path is not None else None
    s = args.split

    run_train(a, b, c, d, s)
