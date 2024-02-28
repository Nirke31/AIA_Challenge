from pathlib import Path
from timeit import default_timer as timer

import torch
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from baseline_submissions.own_model.dataset_manip import load_data, state_change_eval

TRAIN_DATA_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train")
TRAIN_LABEL_PATH = Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")

TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42



df = load_data(TRAIN_DATA_PATH, TRAIN_LABEL_PATH, amount=50)
print("Dataset loaded")
object_ids = df['ObjectID'].unique()
train_ids, test_ids = train_test_split(object_ids,
                                       test_size=1 - TRAIN_TEST_RATIO,
                                       random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=5, class_weight="balanced")
train_data = df.loc[train_ids]
test_data = df.loc[test_ids]

param_grid = {
    'n_estimators': [100, 200, 250, 400],
    'criterion': ['gini', 'entropy']
}
FEATURES = ["Eccentricity",
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
            "Vz (m/s)"]

# f2_scorer = make_scorer(fbeta_score, beta=2)
# cv_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring=f2_scorer,
#                       n_jobs=5, verbose=4, refit=False)
# start_time = timer()
# cv_rfc.fit(df[FEATURES], df["EW"])
# print(f"Took: {timer() - start_time:.3f} seconds")
# print(cv_rfc.best_params_)
# print(cv_rfc.best_score_)
# print(cv_rfc.cv_results_)

print("Fitting...")
start_time = timer()
rf = load("state_classifier_full_job.joblib")
# rf.fit(train_data[FEATURES], train_data["EW"])
print(f"Took: {timer() - start_time:.3f} seconds")
# Write classifier to disk
dump(rf, "state_classifier.joblib")

print("Predicting...")
train_data["PREDICTED"] = rf.predict(train_data[FEATURES])
test_data["PREDICTED"] = rf.predict(test_data[FEATURES])

print("TRAIN RESULTS:")
total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(train_data["PREDICTED"].to_numpy()),
                                                           torch.tensor(train_data["EW"].to_numpy()))
precision = total_tp / (total_tp + total_fp) \
    if (total_tp + total_fp) != 0 else 0
recall = total_tp / (total_tp + total_fn) \
    if (total_tp + total_fn) != 0 else 0
f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0

print(f"Total TPs: {total_tp}")
print(f"Total FPs: {total_fp}")
print(f"Total FNs: {total_fn}")
print(f"Total TNs: {total_tn}")
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F2: {f2:.2f}')

print("TEST RESULTS:")
total_tp, total_fp, total_tn, total_fn = state_change_eval(torch.tensor(test_data["PREDICTED"].to_numpy()),
                                                           torch.tensor(test_data["EW"].to_numpy()))
precision = total_tp / (total_tp + total_fp) \
    if (total_tp + total_fp) != 0 else 0
recall = total_tp / (total_tp + total_fn) \
    if (total_tp + total_fn) != 0 else 0
f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
    if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
print(f"Total TPs: {total_tp}")
print(f"Total FPs: {total_fp}")
print(f"Total FNs: {total_fn}")
print(f"Total TNs: {total_tn}")
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F2: {f2:.2f}')

# plot feature importance
# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=FEATURES)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

