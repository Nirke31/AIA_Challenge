import warnings
from pathlib import Path
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import math

from typing import Tuple, Dict


class MyDataset(IterableDataset):
    def __init__(self, data: pd.DataFrame):
        super(MyDataset).__init__()
        assert data.empty is False, 'Input data is empty'

        # should be sorted but lets just be sure
        data.sort_index(inplace=True)

        # get unique ObjectIDs
        self.first_level_values = data.index.get_level_values(0).unique()

        # separate train src and target data
        tgt_labels = "EW/NS"
        self.src_df = data.drop(labels=tgt_labels, axis=1)
        self.tgt_df = data[tgt_labels].astype('category')

        # create target dict from string label to integer cuz torch.Tensor needs integers/floats
        self.tgt_dict = dict(enumerate(self.tgt_df.cat.categories))

        # convert categorical tgt to numerical tgt, can be translated back with the dicts above
        self.tgt_df = self.tgt_df.cat.codes

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
                           torch.from_numpy(self.tgt_df.loc[objectID].values))

        return yield_time_series(iter_start, iter_end)

    def __len__(self):
        return self.first_level_values.size


def load_data(data_location: str, label_location: str) -> pd.DataFrame:
    data_path = Path(data_location).glob('*.csv')
    label_path = Path(label_location).glob('*.csv')
    # Check if data_location is empty
    if not data_path:
        raise ValueError(f'No csv files found in {data_path}')
    if not label_path:
        raise ValueError(f'No csv file found in {label_path}')

    out_df = pd.DataFrame()  # all data

    labels = pd.read_csv(label_location)  # ObjectID,TimeIndex,Direction,Node,Type

    # Load out_df
    for i, data_file in enumerate(data_path):
        if i == 3:
            break

        datatypes = {"Timestamp": "str", "Eccentricity": "float32", "Semimajor Axis (m)": "float32",
                     "Inclination (deg)": "float32", "RAAN (deg)": "float32", "Argument of Periapsis (deg)": "float32",
                     "True Anomaly (deg)": "float32", "Latitude (deg)": "float32", "Longitude (deg)": "float32",
                     "Altitude (m)": "float32", "X (m)": "float32", "Y (m)": "float32", "Z (m)": "float32",
                     "Vx (m/s)": "float32", "Vy (m/s)": "float32", "Vz (m/s)": "float32"}
        data_df = pd.read_csv(data_file, dtype=datatypes)
        data_df['ObjectID'] = int(data_file.stem)  # csv is named after its objectID/other way round
        data_df['TimeIndex'] = range(len(data_df))
        data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'])  # TODO: convert to posix float?
        data_df.drop(labels='Timestamp', axis=1, inplace=True)  # for now lets just drop it

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
    out_df.drop(labels=['ObjectID', 'TimeIndex', 'EW', 'NS'], axis=1, inplace=True)
    out_df.sort_index(inplace=True)

    return out_df


if __name__ == "__main__":
    train_data_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train"
    train_label_str = "//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train_labels.csv"

    # pd.set_option('display.width', 400)
    # pd.set_option('display.max_columns', None)

    data_df = load_data(train_data_str, train_label_str)

    ds = MyDataset(data_df)
    print(ds.tgt_dict)
    for x, y in ds:
        print(y)
    dl = torch.utils.data.DataLoader(ds, num_workers=0)

    # data_df = pd.read_csv("//wsl$/Ubuntu/home/backwelle/splid-devkit/dataset/phase_1_v2/train/1.csv")
    # data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'])  # convert to posix float?
