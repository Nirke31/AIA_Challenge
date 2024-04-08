import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from timeit import default_timer as timer
from joblib import dump, load

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_manip import load_data, split_train_test, pad_sequence_vec, SubmissionChangePointDataset
from sklearn.preprocessing import StandardScaler
from baseline_submissions.evaluation import NodeDetectionEvaluator

import lightning as L

from statsmodels.tsa.seasonal import seasonal_decompose

from own_model.src.multiScale1DResNet import SimpleResNet
from own_model.src.myModel import LitChangePointClassifier, LitClassifier

train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train"
train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v3/train_labels.csv"

# print("Load whole dataset")
# load1 = timer()
# whole_df = load_data(train_data_str, train_label_str, 99999999)
# time_normal_load = timer() - load1
#
# print("Store...")
# whole_df.to_csv('./dataset/all_data.csv', index=False)
# print("Load it again")
# load2 = timer()
# print(load_data(train_data_str, train_label_str))
# time_all_load = timer() - load2
#
# print(f"First load: {time_normal_load}, Second load: {time_all_load}")

# df = load_data(train_data_str, train_label_str, 3)
#
# split_train_test(df)

# labels = pd.read_csv('./dataset/all_data.csv', usecols=['EW/NS'])
#
# unique_df: pd.DataFrame = labels.drop_duplicates()
#
# unique_df = unique_df.reset_index()
# d = unique_df.to_dict()["EW/NS"]
#
# print(d)
# print(len(d))

# seq_vec = torch.tensor([0, 0])
#
# test = [torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[1, 2]])]
#
# print(test)
# print(pad_sequence_vec(test, seq_vec))

# df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# df = df.sort_values(by=["ObjectID"])
# print(df[df["ObjectID"] == 20])

# df = load_data(Path(train_data_str), Path(train_label_str), 3)
# df = df.loc[1335]
# X = df.drop(["EW", "NS", "ObjectID", "TimeIndex"], axis=1).to_numpy(dtype=np.float64)
#
#
# result = changeforest(X, "random_forest", "bs")
# print(result)
# result.plot().show()
# time.sleep(10)
#


# Using FFT not really benefitical it seems like.
# df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train/178.csv")
# df['Timestamp'] = (pd.to_datetime(df['Timestamp'])).apply(lambda x: x.timestamp()).astype('float32')
# # df = df[["Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)", "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)"]]
# df = df[["Semimajor Axis (m)", "Altitude (m)"]]
#
# # Apply FFT
# test = df.to_numpy()
# print(test.shape)
# fft_result = np.fft.rfft(test, axis=0, norm="ortho")  # Applying FFT along the Sequence length axis
#
# # Extract Magnitude and Phase
# magnitude = np.abs(fft_result)
# phase = np.angle(fft_result)
# magnitude[:10, :] = 0
#
# # Scaling each feature of the magnitude to a value between 0 and 1
# min_vals = np.min(magnitude, axis=1, keepdims=True)
# max_vals = np.max(magnitude, axis=1, keepdims=True)
# max_vals[min_vals == max_vals] = 1 + min_vals[min_vals == max_vals]
# scaled_magnitude = (magnitude - min_vals) / (max_vals - min_vals)
# print(scaled_magnitude.shape)
# plt.plot(magnitude)
# plt.show()

# unique node and type combinations for reconstruction from type label
# df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# df.index = pd.MultiIndex.from_frame(df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
# df = df.drop(["ObjectID", "TimeIndex"], axis=1)
# df_EW = df[df["Direction"] == "EW"]
# df_EW = df_EW.drop_duplicates()
# print(df_EW)  # 'Node' != SS

# from this follows. First change point at position 27+sd6.l
# df = df.drop("TimeIndex", axis=1)
# df = df.loc[df.loc[:, "Node"] != "SS"]
# df.sort_values(by=["TimeIndex"], inplace=True)
# print(df)

# labels = pd.read_csv(train_label_str)
#
# eval = NodeDetectionEvaluator(labels, labels, tolerance=6)
# object_ids = labels["ObjectID"].unique().tolist()
# print(object_ids)
#
# for id in object_ids:
#     eval.plot(id)

# df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# num_all = 1900
# df = df[df["Node"] != "SS"]
# df = df[df["Node"] != "ES"]
# uniqueOBIS = df["ObjectID"].unique()
# uniqueEW = df.loc[df["Direction"] == "EW", :].loc[:, "ObjectID"].unique()
# uniqueNS = df.loc[df["Direction"] == "NS", :].loc[:, "ObjectID"].unique()
# print(len(uniqueOBIS))
# print(len(uniqueEW))
# print(len(uniqueNS))
# print(num_all/len(uniqueOBIS))
# print(num_all/len(uniqueEW))
# print(num_all/len(uniqueNS))

# df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
# df = df[df["Node"] != "ES"]
# df = df[df["Node"] != "SS"]
# df = df[df["Direction"] == "NS"]
# df = df["TimeIndex"].value_counts()
# df.sort_index(inplace=True)
# df.plot(kind='bar')
# plt.show()


# LOOKING AT EW
# for id in range(1151, 2100, 1):
#     data_df = pd.read_csv(train_data_str + f"/{id}.csv")
#     labels = pd.read_csv(train_label_str)
#     data_df.drop(labels='Timestamp', axis=1, inplace=True)
#     data_df['TimeIndex'] = range(len(data_df))
#
#     # Add EW and NS nodes to data. They are extracted from the labels and converted to integers
#     ground_truth_object = labels[labels['ObjectID'] == id].copy()
#     ground_truth_object.drop(labels='ObjectID', axis=1, inplace=True)
#     # Separate the 'EW' and 'NS' types in the ground truth
#     ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
#     ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()
#
#     # Create 'EW' and 'NS' labels and fill 'unknown' values
#     ground_truth_EW['EW'] = 1.0
#     ground_truth_NS['NS'] = 1.0
#
#     ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#     ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#
#     # Merge the input data with the ground truth
#     merged_df = pd.merge(data_df,
#                          ground_truth_EW.sort_values('TimeIndex'),
#                          on=['TimeIndex'],
#                          how='left')
#     merged_df = pd.merge_ordered(merged_df,
#                                  ground_truth_NS.sort_values('TimeIndex'),
#                                  on=['TimeIndex'],
#                                  how='left')
#
#     merged_df['EW_org'] = merged_df['EW']
#
#     indices = merged_df[merged_df['EW'] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer <= seq_len else seq_len
#         merged_df.loc[start:end, "EW"] = 1
#
#     indices = merged_df[merged_df["NS"] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer <= seq_len else seq_len
#         merged_df.loc[start:end, "NS"] = 1
#     # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
#     merged_df['EW'].fillna(0.0, inplace=True)
#     merged_df['EW_org'].fillna(0.0, inplace=True)
#     merged_df['NS'].fillna(0.0, inplace=True)
#     merged_df.drop(["TimeIndex"], axis=1,  # , "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"
#                    inplace=True)
#     features = [
#         # "Eccentricity",
#         # "Semimajor Axis (m)",
#         "Inclination (deg)",
#         "RAAN (deg)",
#         "Argument of Periapsis (deg)",
#         "True Anomaly (deg)",
#         # "Latitude (deg)",
#         "Longitude (deg)",
#         # "Altitude (m)",  # This is just first div of longitude?
#     ]
#     engineered_features_ew = {
#         # ("var", lambda x: x.rolling(window=window_size).var()):
#         #     ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#         ("std", lambda x: x.rolling(window=6).std()):
#             ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],
#         # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
#         # ("skew", lambda x: x.rolling(window=window_size).skew()):
#         #     ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
#         # ("kurt", lambda x: x.rolling(window=window_size).kurt()):
#         #     ["Eccentricity"],  # , "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"
#         # ("sem", lambda x: x.rolling(window=window_size).sem()):
#         #     ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#     }
#     for (math_type, lambda_fnc), feature_list in engineered_features_ew.items():
#         for feature in feature_list:
#             new_feature_name = feature + "_" + math_type + "_EW"
#
#             merged_df[new_feature_name] = lambda_fnc(merged_df[feature])
#             features.append(new_feature_name)
#
#     merged_df["Longitude_diff"] = merged_df["Longitude (deg)"].diff()
#
#     merged_df = merged_df.bfill()
#
#     plt.show()
#     rf = load("../trained_model/state_classifier_EW.joblib")
#     merged_df["PREDICTED"] = rf.predict(merged_df[features])
#     merged_df["PREDICTED_sum"] = merged_df["PREDICTED"].rolling(5, center=True).sum()
#     merged_df["PREDICTED_CLEAN"] = 0
#     merged_df.loc[merged_df["PREDICTED_sum"] >= 5, "PREDICTED_CLEAN"] = 1
#
#     merged_df = pd.DataFrame(StandardScaler().fit_transform(merged_df), index=merged_df.index,
#                              columns=merged_df.columns)
#     test = merged_df[features + ["EW", "EW_org", "PREDICTED", "PREDICTED_sum", "PREDICTED_CLEAN"]]
#     test.plot(subplots=True, title=id)
#     plt.show()

# für postprocessing interessant: 110, 103, 140-160+ total overprediction, 145 weird EW position,

# LOOKING AT NS
# # same for NS
# for id in range(113, 2100, 1):
#     data_df = pd.read_csv(train_data_str + f"/{id}.csv")
#     labels = pd.read_csv(train_label_str)
#     data_df.drop(labels='Timestamp', axis=1, inplace=True)
#     data_df['TimeIndex'] = range(len(data_df))
#
#     # Add EW and NS nodes to data. They are extracted from the labels and converted to integers
#     ground_truth_object = labels[labels['ObjectID'] == id].copy()
#     ground_truth_object.drop(labels='ObjectID', axis=1, inplace=True)
#     # Separate the 'EW' and 'NS' types in the ground truth
#     ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
#     ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()
#
#     # Create 'EW' and 'NS' labels and fill 'unknown' values
#     ground_truth_EW['EW'] = 1.0
#     ground_truth_NS['NS'] = 1.0
#
#     ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#     ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#
#     # Merge the input data with the ground truth
#     merged_df = pd.merge(data_df,
#                          ground_truth_EW.sort_values('TimeIndex'),
#                          on=['TimeIndex'],
#                          how='left')
#     merged_df = pd.merge_ordered(merged_df,
#                                  ground_truth_NS.sort_values('TimeIndex'),
#                                  on=['TimeIndex'],
#                                  how='left')
#
#     merged_df['NS_org'] = merged_df['NS']
#
#     indices = merged_df[merged_df['EW'] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer <= seq_len else seq_len
#         merged_df.loc[start:end, "EW"] = 1
#
#     indices = merged_df[merged_df["NS"] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer <= seq_len else seq_len
#         merged_df.loc[start:end, "NS"] = 1
#     # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
#     merged_df['EW'].fillna(0.0, inplace=True)
#     merged_df['NS'].fillna(0.0, inplace=True)
#     merged_df['NS_org'].fillna(0.0, inplace=True)
#     merged_df.drop(["TimeIndex"], axis=1,  # , "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"
#                    inplace=True)
#     features = [
#         # "Eccentricity",
#         # "Semimajor Axis (m)",
#         "Inclination (deg)",
#         "RAAN (deg)",
#         "Argument of Periapsis (deg)",
#         "True Anomaly (deg)",
#         "Latitude (deg)",
#         "Longitude (deg)",
#         # "Altitude (m)",  # This is just first div of longitude?
#         "X (m)",
#         "Y (m)",
#         "Z (m)",
#         "Vx (m/s)",
#         "Vy (m/s)",
#         "Vz (m/s)"
#     ]
#     engineered_features_ns = {
#         # ("var", lambda x: x.rolling(window=window_size).var()):
#         #     ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#         ("std", lambda x: x.rolling(window=6).std()):
#             ["Semimajor Axis (m)", "Latitude (deg)", "Vz (m/s)", "Z (m)", "RAAN (deg)", "Inclination (deg)"],
#         # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
#         # ("skew", lambda x: x.rolling(window=window_size).skew()):
#         #     ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
#         # ("kurt", lambda x: x.rolling(window=window_size).kurt()):
#         #     ["Eccentricity"],  # , "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"
#         # ("sem", lambda x: x.rolling(window=window_size).sem()):
#         #     ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#     }
#     for (math_type, lambda_fnc), feature_list in engineered_features_ns.items():
#         for feature in feature_list:
#             new_feature_name = feature + "_" + math_type + "_NS"
#
#             merged_df[new_feature_name] = lambda_fnc(merged_df[feature])
#             features.append(new_feature_name)
#
#     merged_df = merged_df.bfill()
#
#     merged_df.plot(subplots=True, title=id)
#     plt.show()
# rf = load("../trained_model/state_classifier.joblib")
# merged_df["PREDICTED"] = rf.predict(merged_df[features])
# merged_df["PREDICTED_sum"] = merged_df["PREDICTED"].rolling(5, center=True).sum()
# merged_df["PREDICTED_CLEAN"] = 0
# merged_df.loc[merged_df["PREDICTED_sum"] >= 5, "PREDICTED_CLEAN"] = 1

# merged_df = pd.DataFrame(StandardScaler().fit_transform(merged_df), index=merged_df.index,
#                          columns=merged_df.columns)
# test = merged_df[features + ["EW", "EW_org", "PREDICTED", "PREDICTED_sum", "PREDICTED_CLEAN"]]
# test.plot(subplots=True, title=id)
# plt.show()
#
# def add_lag_features(df: pd.DataFrame, feature_cols: List[str], lag_steps: int):
#     new_columns = pd.DataFrame({f"{col}_lag{i}": df[col].shift(i * 3)
#                                 for i in range(1, lag_steps + 1)
#                                 for col in feature_cols}, index=df.index)
#     new_columns_neg = pd.DataFrame({f"{col}_lag-{i}": df[col].shift(i * -3)
#                                     for i in range(1, lag_steps + 1)
#                                     for col in feature_cols}, index=df.index)
#     new_df = pd.concat([new_columns, new_columns_neg], axis=1)
#     # basic features were maybe already added, therefore check if these coloumns already exist and don't add them
#     new_df = new_df[new_df.columns.difference(df.columns)]
#     df_out = pd.concat([df, new_df], axis=1)
#     features_out = feature_cols + new_columns.columns.tolist() + new_columns_neg.columns.to_list()
#     # fill nans
#     df_out = df_out.bfill()
#     df_out = df_out.ffill()
#     return df_out, features_out
#
#
# LOOKING AT BOTH
# ground_truth = pd.read_csv("../../dataset/phase_1_v3/train_labels.csv")
# own = pd.read_csv("../../submission/submission_91percent.csv")
# nodeEval = NodeDetectionEvaluator(ground_truth, own, tolerance=6)
# for id in range(1059, 2100, 1):
#     data_df = pd.read_csv(train_data_str + f"/{id}.csv")
#     # data_df = data_df.set_index("Timestamp")
#     data_df.drop("Timestamp", axis=1, inplace=True)
#     times = pd.date_range('2020-01-01', periods=data_df.shape[0], freq='M')
#     data_df.index = times
#
#     for x in ["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)",
#               "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)", "Altitude (m)"]:
#
#         decompose_result = seasonal_decompose(data_df[x], period=12, model="additive")
#         decompose_result.plot()
#         plt.show()

#     labels = pd.read_csv(train_label_str)
#     data_df.drop(labels='Timestamp', axis=1, inplace=True)
#     data_df['TimeIndex'] = range(len(data_df))
#
#     # Add EW and NS nodes to data. They are extracted from the labels and converted to integers
#     ground_truth_object = labels[labels['ObjectID'] == id].copy()
#     ground_truth_object.drop(labels='ObjectID', axis=1, inplace=True)
#     # Separate the 'EW' and 'NS' types in the ground truth
#     ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
#     ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()
#
#     # Create 'EW' and 'NS' labels and fill 'unknown' values
#     ground_truth_EW['EW'] = 1.0
#     ground_truth_NS['NS'] = 1.0
#
#     ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#     ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
#
#     # Merge the input data with the ground truth
#     merged_df = pd.merge(data_df,
#                          ground_truth_EW.sort_values('TimeIndex'),
#                          on=['TimeIndex'],
#                          how='left')
#     merged_df = pd.merge_ordered(merged_df,
#                                  ground_truth_NS.sort_values('TimeIndex'),
#                                  on=['TimeIndex'],
#                                  how='left')
#
#     merged_df['NS_org'] = merged_df['NS']
#     merged_df['EW_org'] = merged_df['EW']
#
#     indices = merged_df[merged_df['EW'] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer < seq_len else seq_len
#         assert end - start == 4
#         merged_df.loc[start:end, "EW"] = 1
#
#     indices = merged_df[merged_df["NS"] == 1].index
#     seq_len = len(data_df)
#     for idx in indices[1:]:
#         puffer = 2
#         start = idx - puffer if idx - puffer >= 0 else 0
#         end = idx + puffer if idx + puffer < seq_len else seq_len
#         assert end - start == 4
#         merged_df.loc[start:end, "NS"] = 1
#     # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
#     merged_df['EW'].fillna(0.0, inplace=True)
#     merged_df['NS'].fillna(0.0, inplace=True)
#     merged_df['NS_org'].fillna(0.0, inplace=True)
#     merged_df['EW_org'].fillna(0.0, inplace=True)
#     merged_df.drop(["TimeIndex"], axis=1,  # , "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"
#                    inplace=True)
#     features_ns = [
#         # "Eccentricity",
#         # "Semimajor Axis (m)",
#         "Inclination (deg)",
#         "RAAN (deg)",
#         # "Argument of Periapsis (deg)",
#         # "True Anomaly (deg)",
#         # "Latitude (deg)",
#         "Longitude (deg)",
#     ]
#     features_ew = [
#         # "Eccentricity",
#         # "Semimajor Axis (m)",
#         # "Inclination (deg)",
#         # "RAAN (deg)",
#         # '"Argument of Periapsis (deg)",
#         # "True Anomaly (deg)",
#         # "Latitude (deg)",
#         "Longitude (deg)",
#         # "Altitude (m)",  # This is just first div of longitude?
#     ]
#     features_all = [
#         "Eccentricity",
#         "Semimajor Axis (m)",
#         "Inclination (deg)",
#         "RAAN (deg)",
#         "Argument of Periapsis (deg)",
#         "True Anomaly (deg)",
#         "Latitude (deg)",
#         "Longitude (deg)",
#         "Altitude (m)"
#     ]
#     DEG_FEATURES = [
#         "Inclination (deg)",
#         "RAAN (deg)",
#         "Argument of Periapsis (deg)",
#         "True Anomaly (deg)",
#         "Latitude (deg)",
#         "Longitude (deg)",
#     ]
#
#     # unwrap
#     merged_df[DEG_FEATURES] = np.unwrap(np.deg2rad(merged_df[DEG_FEATURES]))
#
#     engineered_features_ew = {
#         ("std", lambda x: x.rolling(window=6).std()):
#             ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity"],
#     }
#     for (math_type, lambda_fnc), feature_list in engineered_features_ew.items():
#         for feature in feature_list:
#             new_feature_name = feature + "_" + math_type + "_EW"
#
#             merged_df[new_feature_name] = lambda_fnc(merged_df[feature]).bfill()
#             features_all.append(new_feature_name)
#             features_ew.append(new_feature_name)
#
#     engineered_features_ns = {
#         ("std", lambda x: x.rolling(window=6).std()):
#             ["Semimajor Axis (m)", "Altitude (m)"]
#     }
#     for (math_type, lambda_fnc), feature_list in engineered_features_ns.items():
#         for feature in feature_list:
#             new_feature_name = feature + "_" + math_type + "_NS"
#
#             merged_df[new_feature_name] = lambda_fnc(merged_df[feature]).bfill()
#             # features_all.append(new_feature_name)
#             features_ns.append(new_feature_name)
#
#     merged_df, features_ew = add_lag_features(merged_df, features_ew, lag_steps=8)
#     merged_df, features_ns = add_lag_features(merged_df, features_ns, lag_steps=8)
#
#     merged_df = merged_df.bfill()
#
#     # adding smoothing because of some FPs
#     new_feature_name = "Inclination (deg)" + "_" + "std"
#     merged_df[new_feature_name] = merged_df["Inclination (deg)"].rolling(window=6).std()
#     merged_df[new_feature_name + "smoothed_1"] = (merged_df["Inclination (deg)"])[::-1].ewm(span=100,
#                                                                                             adjust=True).sum()[::-1]
#     merged_df[new_feature_name + "smoothed_2"] = merged_df["Inclination (deg)"].ewm(span=100, adjust=True).sum()
#     merged_df.bfill(inplace=True)
#     merged_df.ffill(inplace=True)
#     merged_df[new_feature_name + "_smoothed"] = (merged_df[new_feature_name + "smoothed_1"] +
#                                                  merged_df[new_feature_name + "smoothed_2"]) / 2
#     features_ew.append(new_feature_name)
#     features_ns.append(new_feature_name)
#     features_ew.append(new_feature_name + "_smoothed")
#     features_ns.append(new_feature_name + "_smoothed")
#
#     rf = load("../trained_model/state_classifier_EW_HistBoosting.joblib")
#     merged_df["PREDICTED_EW"] = rf.predict(merged_df[features_ew])
#     merged_df["PREDICTED_SUM_EW"] = merged_df["PREDICTED_EW"].rolling(5, center=True).sum()
#     merged_df["PREDICTED_CLEAN_EW"] = 0
#     merged_df.loc[merged_df["PREDICTED_SUM_EW"] >= 5, "PREDICTED_CLEAN_EW"] = 1
#     rf = load("../trained_model/state_classifier_NS_HistBoosting.joblib")
#     merged_df["PREDICTED_NS"] = rf.predict(merged_df[features_ns])
#     merged_df["PREDICTED_SUM_NS"] = merged_df["PREDICTED_NS"].rolling(5, center=True).sum()
#     merged_df["PREDICTED_CLEAN_NS"] = 0
#     merged_df.loc[merged_df["PREDICTED_SUM_NS"] >= 5, "PREDICTED_CLEAN_NS"] = 1
#
#     # all_stuff = []
#     # for feature in ["Eccentricity", "Semimajor Axis (m)","Inclination (deg)","RAAN (deg)","Argument of Periapsis (deg)","True Anomaly (deg)","Latitude (deg)","Longitude (deg)","Altitude (m)"]:  # ["Semimajor Axis (m)", "Altitude (m)", "Eccentricity", "RAAN (deg)", "Inclination (deg)"]
#     #     new_feature_name = feature + "_" + "acrr" + "_NS"
#     #     new_stuff: pd.Series = merged_df[feature].rolling(window=24).apply(lambda y: y.autocorr(), raw=False)
#     #     new_stuff.name = new_feature_name
#     #     all_stuff.append(new_stuff)
#     #     features_all.append(new_feature_name)
#     #     features_ns.append(new_feature_name)
#     #
#     # all_new = pd.concat(all_stuff, axis=1)
#     # merged_df = pd.concat([merged_df, all_new], axis=1)
#     # new_feature_name = "Inclination (deg)" + "_" + "std" + "_NS"
#     # merged_df[new_feature_name] = merged_df["Inclination (deg)"].rolling(window=6).std()
#     # merged_df[new_feature_name + "smoothed_1"] = merged_df[new_feature_name][::-1].ewm(span=100, adjust=True).sum()[
#     #                                              ::-1]
#     # merged_df[new_feature_name + "smoothed_2"] = merged_df[new_feature_name].ewm(span=100, adjust=True).sum()
#     # merged_df[new_feature_name + "smoothed"] = (merged_df[new_feature_name + "smoothed_1"] +
#     #                                             merged_df[new_feature_name + "smoothed_2"]) / 2
#     # features_all.append(new_feature_name)
#     # features_all.append(new_feature_name + "smoothed")
#     # merged_df.bfill(inplace=True)
#     # merged_df.ffill(inplace=True)
#
#     # merged_df = pd.DataFrame(StandardScaler().fit_transform(merged_df), index=merged_df.index,
#     #                          columns=merged_df.columns)
#     test = merged_df[features_all + ["EW", "PREDICTED_EW", "PREDICTED_SUM_EW", "PREDICTED_CLEAN_EW"] +
#                      ["NS", "PREDICTED_NS", "PREDICTED_SUM_NS", "PREDICTED_CLEAN_NS"]]
#     test.plot(subplots=True, title=id)
#     nodeEval.plot(id)

# hists
# df = load_data(Path(train_data_str), Path(train_label_str), -1)
# df = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
# df.drop(["EW", "NS", "TimeIndex", "ObjectID"], axis=1, inplace=True)
# df.hist(bins=15)
# plt.show()

# ground_truth = pd.read_csv("../../dataset/phase_1_v3/train_labels.csv")
# own = pd.read_csv("../../submission/submission.csv")
# test = NodeDetectionEvaluator(ground_truth, own, tolerance=6)
# oids = own["ObjectID"].unique().tolist()
# wrong_ids = []
# wrong_FP = []
# wrong_FN = []
# for id in oids:
#     tp, fp, fn, gt_object, p_object = test.evaluate(id)
#     if fp > 0:
#         wrong_FP.append(id)
#     if fn > 0:
#         wrong_FN.append(id)
#     if fp > 0 or fn > 0:
#         wrong_ids.append(id)
#
# print(f"Num wrong time series: {len(wrong_ids)}")
# print(f"Num time series with FN: {len(wrong_FN)}")
# print(f"Num time series with FP: {len(wrong_FP)}")
# print(wrong_ids)
# for id in wrong_ids:
#     test.plot(id)

# precision, recall, f2, rmse = test.score(debug=True)
# print(f'Precision: {precision:.2f}')
# print(f'Recall: {recall:.2f}')
# print(f'F2: {f2:.2f}')
# print(f'RMSE: {rmse:.2f}')
# test.plot(1151)

# # had this in train_classifier
# predict_proba: np.ndarray = rf.predict_proba(test_data[features])[:, 1]
#
# # Step 3: Calculate precision, recall for various thresholds
# precision, recall, thresholds = precision_recall_curve(test_data[DIRECTION], predict_proba)
#
# # Calculate F2 scores for each threshold
# f2_scores = (5 * precision * recall) / ((4 * precision) + recall)
# # Avoid division by zero
# f2_scores = np.nan_to_num(f2_scores)
#
# # Step 4: Find the threshold that maximizes F2 score
# max_f2_index = np.argmax(f2_scores)
# best_threshold = thresholds[max_f2_index]
# best_f2_score = f2_scores[max_f2_index]
#
# print(f"Best threshold: {best_threshold}")
# print(f"Best F2 score: {best_f2_score}")
# test_data["PREDICTED"] = (predict_proba >= best_threshold).astype('int')

lbl_1 = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_label.csv")
lbl_2 = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/test_label.csv")
lbl = pd.concat([lbl_1, lbl_2], axis=0)
lbl.index = lbl["ObjectID"]

object_ids = lbl['ObjectID'].unique()
train_ids, test_ids = train_test_split(object_ids, test_size=0.2, random_state=52, shuffle=True)

lbl_1 = lbl.loc[train_ids]
lbl_2 = lbl.loc[test_ids]

lbl_1.to_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_label_own.csv", index=False)
lbl_2.to_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/test_label_own.csv", index=False)

loaded_dfs = []
for i, data_file in enumerate(Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/training/").glob('*.csv')):
    data_df = pd.read_csv(data_file)
    data_df['ObjectID'] = int(data_file.stem)  # csv is named after its objectID/other way round
    loaded_dfs.append(data_df)

for i, data_file in enumerate(Path("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/test/").glob('*.csv')):
    data_df = pd.read_csv(data_file)
    data_df['ObjectID'] = int(data_file.stem)  # csv is named after its objectID/other way round
    loaded_dfs.append(data_df)

df = pd.concat(loaded_dfs, axis=0)

df.index = df["ObjectID"]
df.drop("ObjectID", axis=1, inplace=True)

df_train = df.loc[train_ids]
df_test = df.loc[test_ids]

for object_id, group in df_train.groupby(level=0, group_keys=False):
    group.to_csv(f"//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/train_own/{int(object_id):04}.csv", index=False)

for object_id, group in df_test.groupby(level=0, group_keys=False):
    group.to_csv(f"//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_2/test_own/{int(object_id):04}.csv", index=False)


