import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import math

from typing import Tuple, Dict


class MyDataset(IterableDataset):
    '''
    Together with dataloader returns torch.tensor src and tgt as well as the objectID(s) of the batch
    '''

    def __init__(self, data: pd.DataFrame):
        super(MyDataset).__init__()
        assert data.empty is False, 'Input data is empty'

        # should be sorted but lets just be sure
        data.sort_index(inplace=True)

        # get unique ObjectIDs
        self.first_level_values = data.index.get_level_values(0).unique()

        # separate train src and target data
        tgt_labels = "EW/NS"
        self.src_df: pd.DataFrame = data.drop(labels=tgt_labels, axis=1)
        self.tgt_df: pd.Series = data[tgt_labels].astype('category')

        # translation dicts. generated from the dataset itself
        self.tgt_dict_int_to_str = {0: 'EW_SS_HK/NS_SS_HK', 1: 'EW_SS_CK/NS_SS_CK', 2: 'EW_SS_EK/NS_SS_CK',
                                    3: 'EW_SS_CK/NS_SS_NK', 4: 'EW_SS_CK/NS_IK_CK', 5: 'EW_SS_HK/NS_SS_NK',
                                    6: 'EW_SS_HK/NS_IK_HK', 7: 'EW_SS_NK/NS_SS_NK', 8: 'EW_AD_NK/NS_SS_NK',
                                    9: 'EW_IK_HK/NS_SS_NK', 10: 'EW_IK_HK/NS_IK_HK', 11: 'EW_IK_HK/NS_IK_CK',
                                    12: 'EW_IK_CK/NS_SS_NK', 13: 'EW_IK_CK/NS_IK_CK', 14: 'EW_ID_NK/NS_ID_NK',
                                    15: 'EW_AD_NK/NS_ID_NK', 16: 'EW_IK_HK/NS_ID_NK', 17: 'EW_ID_NK/NS_SS_NK',
                                    18: 'EW_SS_EK/NS_SS_EK', 19: 'EW_SS_CK/NS_SS_EK', 20: 'EW_SS_EK/NS_SS_NK',
                                    21: 'EW_SS_EK/NS_IK_EK', 22: 'EW_SS_HK/NS_SS_CK', 23: 'EW_SS_EK/NS_ID_NK',
                                    24: 'EW_IK_EK/NS_ID_NK', 25: 'EW_IK_EK/NS_IK_EK', 26: 'EW_IK_CK/NS_ID_NK',
                                    27: 'EW_IK_CK/NS_IK_EK', 28: 'EW_IK_EK/NS_SS_NK', 29: 'EW_IK_EK/NS_IK_CK',
                                    30: 'EW_SS_CK/NS_ID_NK', 31: 'EW_SS_HK/NS_ID_NK'}
        self.tgt_dict_str_to_int = {v: k for k, v in self.tgt_dict_int_to_str.items()}

        # convert categorical tgt to numerical tgt, can be translated back with the dicts above
        self.tgt_df = self.tgt_df.map(self.tgt_dict_str_to_int)
        self.tgt_df = self.tgt_df.astype(dtype='int64')

    def __iter__(self):
        num_object_ids = self.first_level_values.size  # how many ObjectIDs?

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = num_object_ids
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(float(num_object_ids) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, num_object_ids)

        def yield_time_series(iter_start: int, iter_end: int):
            for objectID in self.first_level_values[iter_start:iter_end]:
                # get the current object id and convert to torch tensor
                # warning: numpy to tensor problem cuz numpy is not writable but tensor does not support that
                # -> undefined behaviour on write, but we do not write so its fine (I hope)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    yield (torch.from_numpy(self.src_df.loc[objectID].values),
                           torch.from_numpy(self.tgt_df.loc[objectID].values).unsqueeze(1), objectID)

        return yield_time_series(iter_start, iter_end)

    def __len__(self):
        return self.first_level_values.size


def load_data(data_location: Path, label_location: Path, amount: int = -1) -> pd.DataFrame:
    data_path = data_location.glob('*.csv')
    label_path = label_location.glob('*.csv')
    # Check if data_location is empty
    if not data_path:
        raise ValueError(f'No csv files found in {data_path}')
    if not label_path:
        raise ValueError(f'No csv file found in {label_path}')

    float_size = 'float32'
    datatypes = {"Timestamp": "str", "Eccentricity": float_size, "Semimajor Axis (m)": float_size,
                 "Inclination (deg)": float_size, "RAAN (deg)": float_size,
                 "Argument of Periapsis (deg)": float_size,
                 "True Anomaly (deg)": float_size, "Latitude (deg)": float_size, "Longitude (deg)": float_size,
                 "Altitude (m)": float_size, "X (m)": float_size, "Y (m)": float_size, "Z (m)": float_size,
                 "Vx (m/s)": float_size, "Vy (m/s)": float_size, "Vz (m/s)": float_size}

    if amount < 1:
        # load whole dataset. I pre-safed one already.
        df = pd.read_csv('./dataset/all_data.csv')
        df.index = pd.MultiIndex.from_frame(df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
        return df

    out_df = pd.DataFrame()  # all data
    labels = pd.read_csv(label_location)  # ObjectID,TimeIndex,Direction,Node,Type

    # Load out_df
    for i, data_file in enumerate(data_path):
        if i == amount:
            break

        data_df = pd.read_csv(data_file, dtype=datatypes)
        data_df['ObjectID'] = int(data_file.stem)  # csv is named after its objectID/other way round
        data_df['TimeIndex'] = range(len(data_df))
        # convert timestamp from str to float
        data_df['Timestamp'] = (pd.to_datetime(data_df['Timestamp'])).apply(lambda x: x.timestamp()).astype(float_size)
        # data_df.drop(labels='Timestamp', axis=1, inplace=True)  # for now lets just drop it

        # Add EW and NS nodes to data. They are extracted from the labels and converted to integers

        ground_truth_object = labels[labels['ObjectID'] == data_df['ObjectID'][0]].copy()
        # Separate the 'EW' and 'NS' types in the ground truth
        ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
        ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()

        # Create 'EW' and 'NS' labels and fill 'unknown' values
        ground_truth_EW['EW'] = 'EW_' + ground_truth_EW['Node'] + '_' + ground_truth_EW['Type']
        ground_truth_NS['NS'] = 'NS_' + ground_truth_NS['Node'] + '_' + ground_truth_NS['Type']
        ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
        ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)

        # Merge the input data with the ground truth
        merged_df = pd.merge(data_df,
                             ground_truth_EW.sort_values('TimeIndex'),
                             on=['TimeIndex', 'ObjectID'],
                             how='left')
        merged_df = pd.merge_ordered(merged_df,
                                     ground_truth_NS.sort_values('TimeIndex'),
                                     on=['TimeIndex', 'ObjectID'],
                                     how='left')

        # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
        merged_df['EW'].ffill(inplace=True)
        merged_df['NS'].ffill(inplace=True)

        merged_df['EW/NS'] = merged_df['EW'] + '/' + merged_df['NS']

        out_df = pd.concat([out_df, merged_df])

    out_df_index = pd.MultiIndex.from_frame(out_df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    out_df.index = out_df_index
    # out_df.drop(labels=['ObjectID', 'TimeIndex', 'EW', 'NS'], axis=1, inplace=True)
    out_df.sort_index(inplace=True)

    return out_df


def split_train_test(data: pd.DataFrame, train_test_ration: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # get unique ObjectIDs
    first_level_values: pd.Series = data.index.get_level_values(0).unique().to_series()
    train_indices = first_level_values.sample(frac=train_test_ration, random_state=42)
    test_indices = first_level_values.drop(train_indices.index)

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    return train, test


def convert_tgts_for_eval(pred: torch.Tensor, tgt: torch.Tensor, objectIDs: torch.Tensor,
                          tgt_dict: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # copy input and get on cpu
    tgt = tgt.numpy(force=True)  # 'most likely' a copy, forced of the GPU to cpu
    pred = pred.numpy(force=True)
    objectIDs = objectIDs.numpy(force=True)

    tgt = tgt.squeeze(-1)
    pred = np.argmax(pred, axis=-1)

    # Iterate over all batches
    tgt_vals = []
    pred_vals = []
    batch: np.ndarray
    for batch_pred, batch_tgt, objectID in zip(pred, tgt, objectIDs):
        translate_fnc = np.vectorize(lambda x: tgt_dict[x])

        # translate to strings
        batch_tgt = translate_fnc(batch_tgt)
        batch_pred = translate_fnc(batch_pred)

        # iterate over each entry and translate it
        time_index: int = 0
        for pred, tgt in zip(batch_pred, batch_tgt):
            EW_pred, NS_pred = pred.split('/')
            EW_tgt, NS_tgt = tgt.split('/')
            EW_pred_direction, EW_pred_node, EW_pred_type = EW_pred.split('_')
            NS_pred_direction, NS_pred_node, NS_pred_type = NS_pred.split('_')
            EW_tgt_direction, EW_tgt_node, EW_tgt_type = EW_tgt.split('_')
            NS_tgt_direction, NS_tgt_node, NS_tgt_type = NS_tgt.split('_')

            # store in list which is later translated to pd.dataframe
            pred_vals.append((objectID, time_index, EW_pred_direction, EW_pred_node, EW_pred_type))
            pred_vals.append((objectID, time_index, NS_pred_direction, NS_pred_node, NS_pred_type))
            tgt_vals.append((objectID, time_index, EW_tgt_direction, EW_tgt_node, EW_tgt_type))
            tgt_vals.append((objectID, time_index, NS_tgt_direction, NS_tgt_node, NS_tgt_type))
            time_index += 1

    tgt_df = pd.DataFrame(tgt_vals, columns=['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type'])
    pred_df = pd.DataFrame(pred_vals, columns=['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type'])

    # currently one long dataframe but we only want to have the changes in the dataframe
    # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
    pred_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)
    tgt_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

    # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
    groups_pred = pred_df.groupby(['ObjectID', 'Direction'])
    groups_tgt = tgt_df.groupby(['ObjectID', 'Direction'])
    keep_pred = groups_pred[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)
    keep_tgt = groups_tgt[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)

    # Filter the DataFrame to keep only the rows we're interested in
    keep_pred.index = pred_df.index
    pred_df = pred_df[keep_pred]
    keep_tgt.index = tgt_df.index
    tgt_df = tgt_df[keep_tgt]

    # Reset the index and reorder the columns
    pred_df = pred_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    pred_df = pred_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])
    pred_df = pred_df.reset_index(drop=True)
    tgt_df = tgt_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    tgt_df = tgt_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])
    tgt_df = tgt_df.reset_index(drop=True)

    return pred_df, tgt_df


if __name__ == "__main__":
    train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train"
    train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv"

    # pd.set_option('display.width', 400)
    # pd.set_option('display.max_columns', None)

    data_df = load_data(train_data_str, train_label_str, 3)

    ds = MyDataset(data_df)
    # print(ds.tgt_dict_str_to_int)

    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=1)

    for x, y, ids in dl:
        print(x.shape)
        print(y.shape)
        print(ids)

    # data_df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train/1.csv")
    # data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'])  # convert to posix float?
