from pathlib import Path
from timeit import default_timer as timer

import torch
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from torchmetrics.classification import Accuracy, FBetaScore, Recall, Precision

from own_model.src.dataset_manip import load_first_sample

FEATURES = ["Eccentricity",
            "Semimajor Axis (m)",
            "Inclination (deg)",
            "RAAN (deg)",
            "Argument of Periapsis (deg)",
            "True Anomaly (deg)",
            "Latitude (deg)",
            "Longitude (deg)",
            "Altitude (m)",
            # "X (m)",
            # "Y (m)",
            # "Z (m)",
            # "Vx (m/s)",
            # "Vy (m/s)",
            # "Vz (m/s)"
            ]

TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42

data_df, labels_df = load_first_sample(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=-1)
print("Dataset loaded")
indices = range(labels_df.shape[0])
train_ids, test_ids = train_test_split(indices,
                                       test_size=1 - TRAIN_TEST_RATIO,
                                       random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=12, class_weight="balanced")

train_data = data_df.loc[train_ids, :]
test_data = data_df.loc[test_ids, :]

# get labels and transform v
train_labels = labels_df.loc[train_ids, :]
test_labels = labels_df.loc[test_ids, :]

train_labels_EW = train_labels.loc[:, "EW"]
train_labels_NS = train_labels.loc[:, "NS"]
train_labels_EW = LabelEncoder().fit_transform(train_labels_EW)
train_labels_NS = LabelEncoder().fit_transform(train_labels_NS)
test_labels_EW = test_labels.loc[:, "EW"]
test_labels_NS = test_labels.loc[:, "NS"]
test_labels_EW = LabelEncoder().fit_transform(test_labels_EW)
test_labels_NS = LabelEncoder().fit_transform(test_labels_NS)

print("Fitting...")
start_time = timer()
# rf = load("state_classifier_full_job.joblib")
rf.fit(train_data, train_labels_EW)
print(f"Took: {timer() - start_time:.3f} seconds")
# Write classifier to disk
# dump(rf, "first_sample_classifier.joblib")

print("Predicting...")
train_data_pred = rf.predict(train_data)
test_data_pred = rf.predict(test_data)

# EVAL
train_data_pred = torch.tensor(train_data_pred)
train_labels_EW = torch.tensor(train_labels_EW)
acc_train = Accuracy("multiclass", num_classes=5)(train_data_pred, train_labels_EW)
recall_train = Recall("multiclass", num_classes=5)(train_data_pred, train_labels_EW)
precision_train = Precision("multiclass", num_classes=5)(train_data_pred, train_labels_EW)
f2_train = FBetaScore("multiclass", beta=2.0, num_classes=5)(train_data_pred, train_labels_EW)
print(f'Training Results')
print(f'Accuracy: {acc_train:.2f}')
print(f'Precision: {precision_train:.2f}')
print(f'Recall: {recall_train:.2f}')
print(f'F2: {f2_train:.2f}')

test_data_pred = torch.tensor(test_data_pred)
test_labels_EW = torch.tensor(test_labels_EW)
acc_test = Accuracy("multiclass", num_classes=5)(test_data_pred, test_labels_EW)
recall_test = Recall("multiclass", num_classes=5)(test_data_pred, test_labels_EW)
precision_test = Precision("multiclass", num_classes=5)(test_data_pred, test_labels_EW)
f2_test = FBetaScore("multiclass", beta=2.0, num_classes=5)(test_data_pred, test_labels_EW)
print(f'Test Results')
print(f'Accuracy: {acc_test:.2f}')
print(f'Precision: {precision_test:.2f}')
print(f'Recall: {recall_test:.2f}')
print(f'F2: {f2_test:.2f}')
