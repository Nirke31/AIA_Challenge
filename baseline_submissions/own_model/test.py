import time

import pandas as pd
import torch
from timeit import default_timer as timer
from torch import Tensor
import torch.nn as nn
from dataset_manip import load_data, split_train_test, pad_sequence_vec

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
