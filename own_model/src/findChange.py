import math
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
from own_model.src.myModel import LitChangePointClassifier
from baseline_submissions.evaluation import NodeDetectionEvaluator

TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

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

BASE_FEATURES_NS = ["Eccentricity",
                    "Semimajor Axis (m)",
                    "Inclination (deg)",
                    "RAAN (deg)",
                    "Argument of Periapsis (deg)",
                    "True Anomaly (deg)",
                    "Latitude (deg)",
                    "Longitude (deg)",
                    "Altitude (m)",
                    "X (m)",
                    "Y (m)",
                    "Z (m)",
                    "Vx (m/s)",
                    "Vy (m/s)",
                    "Vz (m/s)"
                    ]

TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42
DIRECTION = "NS"
NUM_CSV_SETS = -1

if __name__ == "__main__":
    df: pd.DataFrame = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=NUM_CSV_SETS)
    # manually remove the change point at time index 0. We know that there is a time change so we do not have to try
    # and predict it
    df.loc[df["TimeIndex"] == 0, "EW"] = 0
    df.loc[df["TimeIndex"] == 0, "NS"] = 0
    # df = df.drop(["ObjectID", "TimeIndex"], axis=1)

    # CNN that didnt work so far and actually hangs, maybe actually because no scaling?
    # objectIDs = df.index.get_level_values(0).unique().to_series()
    # objectIDs = objectIDs.sample(frac=1)
    # objectIDs.reset_index(drop=True, inplace=True)
    # idSize = objectIDs.shape[0]
    # objectIDs_train = objectIDs[:math.floor(idSize * 0.6)]
    # objectIDs_val = objectIDs[math.floor(idSize * 0.6): math.floor(idSize * 0.8)]
    # objectIDs_test = objectIDs[math.floor(idSize * 0.8):]
    #
    # data_train = df.loc[(objectIDs_train, slice(1, 2100)), :].copy()
    # data_val = df.loc[(objectIDs_val, slice(1, 2100)), :].copy()
    # data_test = df.loc[(objectIDs_test, slice(1, 2100)), :].copy()
    #
    # # test_df = train_df.copy()  # FOR DEBUGGING ONLY
    # ds_train = MyDataset(data_train)
    # ds_val = MyDataset(data_val)
    # ds_test = MyDataset(data_test)
    # dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=0)
    # dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=0)
    # dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, num_workers=0)
    #
    # model = LitChangePointClassifier(2100, 15)
    # early_stop_callback = EarlyStopping(monitor="val_f2", mode="max", patience=5)
    # trainer = L.Trainer(max_epochs=EPOCHS, enable_progress_bar=True) # ,callbacks=[early_stop_callback], check_val_every_n_epoch=10
    # trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    # trainer.test(model=model, dataloaders=dataloader_test)

    # Isolation forest

    # RF approach
    print("Dataset loaded")
    object_ids = df['ObjectID'].unique()
    train_ids, test_ids = train_test_split(object_ids,
                                           test_size=1 - TRAIN_TEST_RATIO,
                                           random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=14, class_weight="balanced", )

    # features selected based on rf feature importance.
    features = BASE_FEATURES_EW if DIRECTION == "EW" else BASE_FEATURES_NS
    engineered_features_ew = {
        # ("var", lambda x: x.rolling(window=window_size).var()):
        #     ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],
        # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
        # ("skew", lambda x: x.rolling(window=window_size).skew()):
        #     ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
        # ("kurt", lambda x: x.rolling(window=window_size).kurt()):
        #     ["Eccentricity"],  # , "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"
        # ("sem", lambda x: x.rolling(window=window_size).sem()):
        #     ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
    }
    engineered_features_ns = {
        # ("var", lambda x: x.rolling(window=window_size).var()):
        #     ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
        ("std", lambda x: x.rolling(window=window_size).std()):
            ["Semimajor Axis (m)", "Latitude (deg)", "Vz (m/s)", "Z (m)", "RAAN (deg)", "Inclination (deg)"],  # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
        # ("skew", lambda x: x.rolling(window=window_size).skew()):
        #     ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
        # ("kurt", lambda x: x.rolling(window=window_size).kurt()):
        #     ["Eccentricity", "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"],
        # ("sem", lambda x: x.rolling(window=window_size).sem()):
        #     ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
    }

    # FEATURE ENGINEERING
    window_size = 6
    feature_dict = engineered_features_ew if DIRECTION == "EW" else engineered_features_ns
    for (math_type, lambda_fnc), feature_list in feature_dict.items():
        for feature in feature_list:
            new_feature_name = feature + "_" + math_type + "_" + DIRECTION
            # groupby objectIDs, get a feature and then apply rolling window for each objectID, is returned as series
            # and then added back to the DF
            df[new_feature_name] = df.groupby(level=0, group_keys=False)[[feature]].apply(lambda_fnc)
            features.append(new_feature_name)

    # fill beginning of rolling window (NaNs). Shouldn't really matter anyways? Maybe else Median
    df = df.bfill()

    test_data = df.loc[test_ids].copy()
    train_data = df.loc[train_ids].copy()
    # train_data = df.copy()

    # f2_scorer = make_scorer(fbeta_score, beta=2)
    # param_grid = {
    #     'ccp_alpha': [0.0, 0.01, 0.02, 0.04, 0.1],
    #     'min_samples_split': [2, 3, 4],
    #     'criterion': ["gini", "entropy"]
    # }
    # cv_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=f2_scorer,
    #                       n_jobs=16, verbose=4, refit=False)
    # start_time = timer()
    # cv_rfc.fit(df[features], df[DIRECTION])
    # print(f"Took: {timer() - start_time:.3f} seconds")
    # print(cv_rfc.best_params_)
    # print(cv_rfc.best_score_)
    # print(cv_rfc.cv_results_)

    # RF model
    print("Fitting...")
    start_time = timer()
    # rf = load("../trained_model/state_classifier_EW.joblib")
    rf.fit(train_data[features], train_data[DIRECTION])
    print(f"Took: {timer() - start_time:.3f} seconds")
    # Write classifier to disk
    dump(rf, "../trained_model/state_classifier.joblib", compress=0)

    print("Predicting...")
    train_data["PREDICTED"] = rf.predict(train_data[features])
    test_data["PREDICTED"] = rf.predict(test_data[features])

    # post processing
    test_data["PREDICTED_sum"] = test_data["PREDICTED"].rolling(5, center=True).sum()
    test_data["PREDICTED_CLEAN"] = 0
    test_data.loc[test_data["PREDICTED_sum"] >= 5, "PREDICTED_CLEAN"] = 1
    # set manually
    test_data.loc[test_data["TimeIndex"] == 0, "PREDICTED_CLEAN"] = 1

    changepoints = test_data.loc[test_data["PREDICTED_CLEAN"].astype("bool"), ["ObjectID", "TimeIndex"]]
    changepoints["Direction"] = "EW"
    changepoints["Node"] = "EGAL"
    changepoints["Type"] = "EGAL"

    object_ids = changepoints["ObjectID"].unique()

    labels = pd.read_csv(TRAIN_LABEL_PATH)
    labels = labels.loc[labels["ObjectID"].isin(object_ids), :]
    lable_ns = labels[labels["Direction"] == "NS"]
    # Currently not predicting NS therefore just add the correct ones
    changepoints = pd.concat([changepoints, lable_ns])
    changepoints = changepoints.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    eval = NodeDetectionEvaluator(labels, changepoints, tolerance=6)
    precision, recall, f2, rmse = eval.score(debug=True)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')

    eval.plot(object_ids[0])


    # print("TRAIN RESULTS:")
    # total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(train_data["PREDICTED"].to_numpy()),
    #                                                            torch.tensor(train_data[DIRECTION].to_numpy()))
    # precision = total_tp / (total_tp + total_fp) \
    #     if (total_tp + total_fp) != 0 else 0
    # recall = total_tp / (total_tp + total_fn) \
    #     if (total_tp + total_fn) != 0 else 0
    # f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    #     if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
    #
    # print(f"Total TPs: {total_tp}")
    # print(f"Total FPs: {total_fp}")
    # print(f"Total FNs: {total_fn}")
    # print(f"Total TNs: {total_tn}")
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')
    #
    # print("TEST RESULTS:")
    # total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(test_data["PREDICTED"].to_numpy()),
    #                                                            torch.tensor(test_data[DIRECTION].to_numpy()))
    # precision = total_tp / (total_tp + total_fp) \
    #     if (total_tp + total_fp) != 0 else 0
    # recall = total_tp / (total_tp + total_fn) \
    #     if (total_tp + total_fn) != 0 else 0
    # f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    #     if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
    # print(f"Total TPs: {total_tp}")
    # print(f"Total FPs: {total_fp}")
    # print(f"Total FNs: {total_fn}")
    # print(f"Total TNs: {total_tn}")
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')
    # print(f'F2: {f2:.2f}')

    # feature importance with permutation, more robust or smth like that
    # start_time = timer()
    # f2_scorer = make_scorer(fbeta_score, beta=2)
    # result = permutation_importance(rf, test_data[features], test_data[DIRECTION].to_numpy(), n_repeats=10,
    #                                 random_state=RANDOM_STATE, n_jobs=1, scoring=f2_scorer)
    # elapsed_time = timer() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    # forest_importances = pd.Series(result.importances_mean, index=features)
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()
