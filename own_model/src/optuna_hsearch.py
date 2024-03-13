from typing import List

import optuna
from numpy.lib.stride_tricks import sliding_window_view
from optuna import create_study

from sklearn.model_selection import cross_val_score, KFold
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset


def custom_loss_func(ground_truth: np.array, predictions: np.array, **kwargs):
    predicted_sum = np.convolve(predictions, np.ones(5), mode="same")
    predicted = np.zeros(predicted_sum.shape[0])
    predicted[predicted_sum >= 5] = 1

    ground_truth_sum = np.convolve(ground_truth, np.ones(5), mode="same")
    truth = np.zeros(ground_truth_sum.shape[0])
    truth[ground_truth_sum >= 5] = 1

    # f2 score
    f2_score = fbeta_score(truth, predicted, beta=2, zero_division=0.0)
    return f2_score


def add_lag_features(df: pd.DataFrame, feature_cols: List[str], lag_steps: int):
    new_columns = pd.DataFrame({f"{col}_lag{i}": df.groupby(level=0, group_keys=False)[col].shift(i * 3)
                                for i in range(1, lag_steps + 1)
                                for col in feature_cols}, index=df.index)
    new_columns_neg = pd.DataFrame({f"{col}_lag-{i}": df.groupby(level=0, group_keys=False)[col].shift(i * -3)
                                    for i in range(1, lag_steps + 1)
                                    for col in feature_cols}, index=df.index)
    df_out = pd.concat([df, new_columns, new_columns_neg], axis=1)
    features_out = feature_cols + new_columns.columns.tolist() + new_columns_neg.columns.to_list()
    # fill nans
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.bfill())
    df_out = df_out.groupby(level=0, group_keys=False).apply(lambda x: x.ffill())
    return df_out, features_out


def load_stuff():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0
    # RF approach
    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW.copy() if DIRECTION == "EW" else BASE_FEATURES_NS.copy()

    # unwrap
    df[features] = np.unwrap(np.deg2rad(df[features]))

    # FEATURE ENGINEERING
    window_size = 6
    feature_dict = ENGINEERED_FEATURES_EW if DIRECTION == "EW" else ENGINEERED_FEATURES_NS
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_" + DIRECTION
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            features.append(new_feature_name)

    add_lag_features(df, features, lag_steps=8)

    return df, features


def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.5])
    max_iter = trial.suggest_categorical("max_iter", [50, 100, 200, 300, 400])
    max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None, 21, 31, 41])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [20, 25, 30])
    l2_regularization = trial.suggest_categorical("l2_regularization", [0, 0.0001, 0.001, 0.1])

    rf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced", learning_rate=learning_rate,
                                        max_iter=max_iter, early_stopping=False, max_leaf_nodes=max_leaf_nodes,
                                        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization)

    f2_scorer = make_scorer(fbeta_score, beta=2)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score = cross_val_score(rf, df[features], df[DIRECTION], n_jobs=5, scoring=f2_scorer, cv=kf)
    f2_score = score.mean()
    return f2_score


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train")
    TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv")

    BASE_FEATURES_EW = [
        # "Eccentricity",
        # "Semimajor Axis (m)",
        # "Inclination (deg)",
        # "RAAN (deg)",
        # '"Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        "Longitude (deg)",
        # "Altitude (m)",  # This is just first div of longitude?
    ]
    BASE_FEATURES_NS = [
        # "Eccentricity",
        # "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        # "Argument of Periapsis (deg)",
        # "True Anomaly (deg)",
        # "Latitude (deg)",
        "Longitude (deg)",
        # "Altitude (m)",
        # "X (m)",
        # "Y (m)",
        # "Z (m)",
        # "Vx (m/s)",
        # "Vy (m/s)",
        # "Vz (m/s)"
    ]

    ENGINEERED_FEATURES_EW = {
        ("std", lambda x: x.rolling(window=6).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],  # , "RAAN (deg)"
    }
    ENGINEERED_FEATURES_NS = {
        ("std", lambda x: x.rolling(window=6).std()):
            ["Semimajor Axis (m)", "Altitude (m)"],
        # "Semimajor Axis (m)", "Latitude (deg)", "RAAN (deg)", "Inclination (deg)"
    }

    DEG_FEATURES = [
        "Inclination (deg)",
        "RAAN (deg)",
        "Argument of Periapsis (deg)",
        "True Anomaly (deg)",
        "Latitude (deg)",
        "Longitude (deg)"
    ]

    RANDOM_STATE = 42
    DIRECTION = "NS"
    NUM_CSV_SETS = -1

    # Beginning
    df, features = load_stuff()

    study_name = "hSearch_HistGrad_NS"
    db_file_path = "C:/Users/nikla/Desktop/test_db/mystorage.db"
    # storage = optuna.storages.RDBStorage(url=db_file_path, engine_kwargs={"connect_args": {"timeout": 100}})
    sampler = optuna.samplers.CmaEsSampler()
    study = create_study(load_if_exists=True, study_name=study_name, direction="maximize",
                         storage=f'sqlite:///{db_file_path}')
    study.optimize(objective, n_trials=200)
    # dump(study, "study.pkl")
    # Print the optimal hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial)
