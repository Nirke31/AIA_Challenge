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
from baseline_submissions.evaluation import NodeDetectionEvaluator

# https://eval.ai/web/challenges/challenge-page/2164/overview

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
df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv")
df.index = pd.MultiIndex.from_frame(df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
df = df.drop(["ObjectID", "TimeIndex"], axis=1)
df_EW = df[df["Direction"] == "EW"]
df_EW = df_EW.drop_duplicates()
print(df_EW)  # 'Node' != SS

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

