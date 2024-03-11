import optuna
from optuna import create_study

from sklearn.model_selection import cross_val_score
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
import lightning as L
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from own_model.src.dataset_manip import load_data, state_change_eval, MyDataset


def load_stuff():
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0
    # RF approach
    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW.copy() if DIRECTION == "EW" else BASE_FEATURES_NS.copy()
    engineered_features_ew = {
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],
    }
    engineered_features_ns = {
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],
    }

    # unwrap
    df[features] = np.unwrap(np.deg2rad(df[features]))

    # FEATURE ENGINEERING
    window_size = 6
    feature_dict = engineered_features_ew if DIRECTION == "EW" else engineered_features_ns
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_" + DIRECTION
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc).bfill()
            features.append(new_feature_name)

    return df, features


def objective(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200, 400, 800, 1000])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 3, 4, 6, 8])
    ccp_alpha = trial.suggest_categorical('ccp_alpha', [0.0, 0.01, 0.04, 0.08, 0.1, 0.3, 0.6, 1.0])

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=14,
                                class_weight="balanced", min_samples_split=min_samples_split,
                                ccp_alpha=ccp_alpha)
    f2_scorer = make_scorer(fbeta_score, beta=2)

    rf.fit(train_data[features], train_data[DIRECTION])
    predicted = rf.predict(test_data[features])
    f2_score = fbeta_score(test_data[DIRECTION], predicted, beta=2.0)

    # score = cross_val_score(rf, df[features], df["NS"], n_jobs=1, cv=3, scoring=f2_scorer)
    # f2_score = score.mean()
    return f2_score


if __name__ == "__main__":
    TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train")
    TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv")

    BASE_FEATURES_EW = [
        # "Eccentricity",
        # "Semimajor Axis (m)",
        "Inclination (deg)",
        "RAAN (deg)",
        "Argument of Periapsis (deg)",
        "True Anomaly (deg)",
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
        "True Anomaly (deg)",
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

    RANDOM_STATE = 42
    DIRECTION = "NS"
    NUM_CSV_SETS = -1

    # Beginning
    df, features = load_stuff()
    object_ids = df['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids, test_size=0.2, random_state=RANDOM_STATE)
    test_data = df.loc[test_ids].copy()
    train_data = df.loc[train_ids].copy()

    study_name = "hSearch"
    db_file_path = "C:/Users/nikla/Desktop/test_db/mystorage.db"
    # storage = optuna.storages.RDBStorage(url=db_file_path, engine_kwargs={"connect_args": {"timeout": 100}})
    study = create_study(load_if_exists=True, study_name=study_name, direction="maximize",
                         storage=f'sqlite:///{db_file_path}')
    study.optimize(objective, n_trials=100)
    # dump(study, "study.pkl")
    # Print the optimal hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    print('Best params:', study.best_params)
    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial)
