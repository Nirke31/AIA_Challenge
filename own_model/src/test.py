import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timeit import default_timer as timer

from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn as nn
from dataset_manip import load_data, split_train_test, pad_sequence_vec
from changeforest import changeforest
from sklearn.preprocessing import StandardScaler
from baseline_submissions.evaluation import NodeDetectionEvaluator

train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train"
train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv"

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

# for id in range(1000, 1200):
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
#     indecies = merged_df[merged_df['EW'] == 1].index
#     # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
#     merged_df['EW'].fillna(0.0, inplace=True)
#     merged_df['NS'].fillna(0.0, inplace=True)
#     merged_df.drop(["TimeIndex", "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"], axis=1,
#                    inplace=True)
#     engineered_features_ew = {
#         ("var", lambda x: x.rolling(window=6).var()):
#             ["Semimajor Axis (m)"],  # , "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#         ("std", lambda x: x.rolling(window=6).std()):
#             ["Semimajor Axis (m)"],  # "Eccentricity", "Semimajor Axis (m)", "Longitude (deg)", "Altitude (m)"
#         ("skew", lambda x: x.rolling(window=6).skew()):
#             ["Eccentricity"],  # , "Semimajor Axis (m)", "Argument of Periapsis (deg)", "Altitude (m)"
#         ("kurt", lambda x: x.rolling(window=6).kurt()):
#             ["Eccentricity", "Argument of Periapsis (deg)", "Semimajor Axis (m)", "Longitude (deg)"],
#         ("sem", lambda x: x.rolling(window=6).sem()):
#             ["Longitude (deg)"],  # "Eccentricity", "Argument of Periapsis (deg)", "Longitude (deg)", "Altitude (m)"
#     }
#     for (math_type, lambda_fnc), feature_list in engineered_features_ew.items():
#         for feature in feature_list:
#             new_feature_name = feature + "_" + math_type + "_EW"
#
#             merged_df[new_feature_name] = lambda_fnc(merged_df[feature])
#     merged_df = merged_df.bfill()
#
#     merged_df = pd.DataFrame(StandardScaler().fit_transform(merged_df), index=merged_df.index, columns=merged_df.columns)
#     test = merged_df.plot(subplots=True, title=id)
#     plt.show()

df = load_data(Path(train_data_str), Path(train_label_str), -1)
df = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
df.drop(["EW", "NS", "TimeIndex", "ObjectID"], axis=1, inplace=True)
df.hist(bins=15)
plt.show()
