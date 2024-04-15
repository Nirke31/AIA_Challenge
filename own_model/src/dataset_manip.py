import math
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset


class SubmissionWindowDataset(IterableDataset):
    """
    Special dataset for the submission. Takes pandas dataframe as input. It includes a state change column
    (either 1 for change or 0 for no change). Create windows for all time indices that are 1.
    """

    def __init__(self, data: pd.DataFrame, features: List[str], coloum_name: str, window_size: int):
        super().__init__()
        assert data.empty is False, 'Input is empty brother'
        if window_size % 2 == 0:
            raise warnings.warn("I think odd window sizes is needed. Unknown behaviour maybe?")
        self.window_size: int = window_size
        self.features: List[str] = features

        self.data_in: pd.DataFrame = data
        feature_size = len(self.features)

        num_change_points = (self.data_in.loc[:, coloum_name]).astype(dtype="int64").sum()

        # create empty array for storing all window blocks
        # later in __iter__ we only have to iterate over the src and tgt arrays and return them
        # src is size [number of tgts, sequence length of window, feature size]
        self.src = np.empty((num_change_points, self.window_size, feature_size), dtype=np.float32)
        # storing the pandas dataframe of all the row that will be classified
        self.predicted_tgts = self.data_in.loc[
            self.data_in.loc[:, coloum_name].astype(dtype="bool"), ["ObjectID", "TimeIndex"]]

        # translation dicts. generated from the dataset itself
        self.tgt_dict_int_to_str = {0: "NK", 1: "CK", 2: "EK", 3: "HK", 4: "FAKE"}
        self.tgt_dict_str_to_int = {v: k for k, v in self.tgt_dict_int_to_str.items()}

        # create all window sequences
        self._prepare_source(self.data_in.shape[-1])


    def _prepare_source(self, feature_size) -> None:

        data_only_features = self.data_in.loc[:, self.features]
        for i, (row_idx, row) in enumerate(self.predicted_tgts.iterrows()):
            objectID = row["ObjectID"]
            time_index = row["TimeIndex"]
            # The sequence length of a specific objectID. Important to not window out of the sequence
            src_seq_len = self.data_in.loc[objectID].shape[0]

            offset = (self.window_size - 1) / 2
            start_iter = time_index - offset
            end_iter = time_index + offset
            # check out of range
            if start_iter < 0:
                # window would go negative. Therefore, move window upward
                end_iter -= start_iter
                start_iter = 0
            elif end_iter >= src_seq_len:
                # window would go beyond sequence. Therefore, move window down (-1 because pandas)
                start_iter -= (end_iter - src_seq_len) + 1
                end_iter = src_seq_len - 1

            # the end_iter is also returned by pandas because it is accessing via index (?!)
            # my end_iter is therefore one smaller than 'normally'
            assert (end_iter - start_iter) == self.window_size - 1, "Something wrong"

            # push window into src
            self.src[i, :, :] = data_only_features.loc[(objectID, start_iter): (objectID, end_iter)].to_numpy()
        return

    def __iter__(self):
        # reset index, such that we can use a simple range to iterate over the src array and pandas dataframe
        self.predicted_tgts.reset_index(drop=True, inplace=True)

        size = self.src.shape[0]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = size
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(float(size) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, size)

        def yield_time_series():
            for row_idx in range(iter_start, iter_end):
                # warning: numpy to tensor problem cuz numpy is not writable but tensor does not support that
                # -> undefined behaviour on write, but we do not write so its fine (I hope)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    src_array = self.src[row_idx, :, :]
                    yield (torch.from_numpy(src_array),
                           torch.tensor(self.predicted_tgts.loc[row_idx, "ObjectID"]),
                           torch.tensor(self.predicted_tgts.loc[row_idx, "TimeIndex"]))

        return yield_time_series()

    def __len__(self):
        return self.src.shape[0]


class SubmissionChangePointDataset(Dataset):
    """
    Special dataset for submission
    """

    def __init__(self, data: pd.DataFrame, window_size: int):
        """
        Args:
            data: dataframe with all rows and coloumns.
            tgts: Series with all rows
            idx: np array with indices to access the correct rows from data and tgts
            window_size: size of the window
        """
        super().__init__()
        assert data.empty is False, 'Input is empty brother'
        if window_size % 2 == 0:
            raise warnings.warn("I think odd window sizes is needed. Unknown behaviour maybe?")
        self.window_size = window_size
        self.feature_size = data.shape[-1]
        self.offset = (self.window_size - 1) // 2
        self.data_in = data

        self.idx_to_src = np.arange(self.data_in.shape[0])

        # src created here
        self._prepare_source(data.sort_index())

    def _prepare_source(self, data):
        objectIDs = data.index.get_level_values(level=0).unique()
        num_objs = objectIDs.shape[0]

        # inbetween each object and at beginning and end there is zero padding
        rows = data.shape[0] + (num_objs + 1) * self.offset
        cols = self.feature_size
        self.src = np.zeros((rows, cols), dtype=np.float32)

        curr_offset = self.offset
        overall_seq_len = 0
        # start at first timestep and not zero padding
        self.idx_to_src[:] += self.offset
        for _, group in data.groupby(level=0, group_keys=False):
            seq_len = group.shape[0]
            self.src[curr_offset:curr_offset + seq_len, :] = group.to_numpy()

            # group plus zero padding
            curr_offset += seq_len + self.offset
            # update index mapping
            overall_seq_len += seq_len
            self.idx_to_src[overall_seq_len:] += self.offset
        return

    def __getitem__(self, item):
        mapped_idx = self.idx_to_src[item]
        start_idx = mapped_idx - self.offset
        end_idx = mapped_idx + self.offset + 1

        return (torch.from_numpy(self.src[start_idx:end_idx, :]),
                torch.tensor(self.data_in.loc[item, "ObjectID"]),
                torch.tensor(self.data_in.loc[item, "TimeIndex"]))

    def __len__(self):
        return self.data_in.shape[0]


class GetWindowDataset(Dataset):
    """
    input data must already have Multiindex (ObjectId, TimeIndex)
    """

    def __init__(self, data: pd.DataFrame, tgt_index_and_label: pd.DataFrame, window_size: int):
        super().__init__()
        assert data.empty is False, 'Input is empty brother'
        if window_size % 2 == 0:
            raise warnings.warn("I think odd window sizes is needed. Unknown behaviour maybe?")
        self.window_size = window_size

        self.data_in = data
        self.tgt = tgt_index_and_label
        num_tgts = self.tgt.shape[0]
        feature_size = self.data_in.shape[-1]
        # create empty array for storing all window blocks
        # later in __iter__ we only have to iterate over the src and tgt arrays and return them
        # src is size [number of tgts, sequence length of window, feature size]
        self.src = np.empty((num_tgts, self.window_size, feature_size), dtype=np.float32)

        # translation dicts. generated from the dataset itself
        self.tgt_dict_int_to_str = {0: "NK", 1: "CK", 2: "EK", 3: "HK", 4: "FAKE"}
        self.tgt_dict_str_to_int = {v: k for k, v in self.tgt_dict_int_to_str.items()}

        # convert categorical tgt to numerical tgt, can be translated back with the dicts above
        self.tgt.loc[:, "Type"] = self.tgt.loc[:, "Type"].map(self.tgt_dict_str_to_int)
        self.tgt.loc[:, "Type"] = self.tgt.loc[:, "Type"].astype(dtype='int64')

        # create all window sequences
        self._prepare_source(self.data_in.shape[-1])

        # # create extra feature, showing which window is starting at index 0
        # zero_timeIndex_feature = (self.tgt.loc[:, "TimeIndex"] == 0).to_numpy().astype('float32')
        # test = np.expand_dims(zero_timeIndex_feature, axis=(1, 2))
        # test = np.repeat(test, window_size, axis=1)
        # self.src = np.append(self.src, test, axis=-1)

    def _prepare_source(self, feature_size) -> None:
        for i, (row_idx, row) in enumerate(self.tgt.iterrows()):
            objectID = row["ObjectID"]
            time_index = row["TimeIndex"]
            # The sequence length of a specific objectID. Important to not window out of the sequence
            src_seq_len = self.data_in.loc[objectID].shape[0]

            offset = (self.window_size - 1) / 2
            start_iter = time_index - offset
            end_iter = time_index + offset
            # check out of range
            if start_iter < 0:
                # window would go negative. Therefore, move window upward
                end_iter -= start_iter
                start_iter = 0
            elif end_iter >= src_seq_len:
                # window would go beyond sequence. Therefore, move window down (-1 because pandas)
                start_iter -= (end_iter - src_seq_len) + 1
                end_iter = src_seq_len - 1

            # the end_iter is also returned by pandas because it is accessing via index (?!)
            # my end_iter is therefore one smaller than 'normaly'
            assert (end_iter - start_iter) == self.window_size - 1, "Something wrong"

            # push window into src
            self.src[i, :, :] = self.data_in[(objectID, start_iter): (objectID, end_iter)].to_numpy()
        return

    def __getitem__(self, item):
        src_array = self.src[item, :, :]
        tgt_value_int = self.tgt.loc[item, "Type"]
        return (torch.from_numpy(src_array),
                torch.tensor([[tgt_value_int]]),
                self.tgt.loc[item, "ObjectID"],
                self.tgt.loc[item, "TimeIndex"])

    def __len__(self):
        return self.tgt.shape[0]


class ChangePointDataset(Dataset):
    """
        creates one large array with zero padding between objectsIDs
    """

    def __init__(self, data: pd.DataFrame, tgts: pd.Series, window_size: int):
        """
        Args:
            data: dataframe with all rows and coloumns.
            tgts: Series with all rows
            idx: np array with indices to access the correct rows from data and tgts
            window_size: size of the window
        """
        super().__init__()
        assert data.empty is False, 'Input is empty brother'
        if window_size % 2 == 0:
            raise warnings.warn("I think odd window sizes is needed. Unknown behaviour maybe?")
        self.window_size = window_size
        self.feature_size = data.shape[-1]
        self.offset = (self.window_size - 1) // 2

        self.tgt = tgts.to_numpy(dtype=np.float32)
        self.idx_to_src = np.arange(self.tgt.shape[0])

        # src created here
        self._prepare_source(data.sort_index())

    def _prepare_source(self, data):
        objectIDs = data.index.get_level_values(level=0).unique()
        num_objs = objectIDs.shape[0]

        # inbetween each object and at beginning and end there is zero padding
        rows = data.shape[0] + (num_objs + 1) * self.offset
        cols = self.feature_size
        self.src = np.zeros((rows, cols), dtype=np.float32)

        curr_offset = self.offset
        overall_seq_len = 0
        # start at first timestep and not zero padding
        self.idx_to_src[:] += self.offset
        for _, group in data.groupby(level=0, group_keys=False):
            seq_len = group.shape[0]
            self.src[curr_offset:curr_offset + seq_len, :] = group.to_numpy()

            # group plus zero padding
            curr_offset += seq_len + self.offset
            # update index mapping
            overall_seq_len += seq_len
            self.idx_to_src[overall_seq_len:] += self.offset
        return

    def __getitem__(self, item):
        mapped_idx = self.idx_to_src[item]
        start_idx = mapped_idx - self.offset
        end_idx = mapped_idx + self.offset + 1

        return (torch.from_numpy(self.src[start_idx:end_idx, :]),
                torch.tensor(self.tgt[item]).unsqueeze(0).unsqueeze(0))

    # def __getitems__(self, items):
    #     mapped_idxs = self.idx_to_src[items]
    #     start_idx = mapped_idxs - self.offset
    #     end_idx = mapped_idxs + self.offset + 1
    #     return [(torch.from_numpy(self.src[s_idx:e_idx, :]),
    #              torch.tensor(self.tgt[items]).unsqueeze(0).unsqueeze(0))
    #             for mapped_idx, s_idx, e_idx in zip(mapped_idxs, start_idx, end_idx)]

    def __len__(self):
        return self.tgt.shape[0]


class MyDataset(IterableDataset):
    """
    Together with dataloader returns torch.tensor src and tgt as well as the objectID(s) of the batch
    """

    def __init__(self, data: pd.DataFrame):
        super(MyDataset).__init__()
        assert data.empty is False, 'Input data is empty'

        # should be sorted but lets just be sure
        data.sort_index(inplace=True)

        # get unique ObjectIDs
        self.first_level_values = data.index.get_level_values(0).unique()

        # separate train src and target data
        tgt_labels = ["EW", "NS"]
        self.src_df: pd.DataFrame = data.drop(labels=tgt_labels, axis=1)
        # Lets just look at EW for now
        self.tgt_df: pd.Series = data[tgt_labels[0]].astype(dtype='float32')

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
                           torch.from_numpy(self.tgt_df.loc[objectID].values).unsqueeze(1),
                           objectID)

        return yield_time_series(iter_start, iter_end)

    def __len__(self):
        return self.first_level_values.size


def load_data_window_ready(data_location: Path, label_location: Path, amount: int = -1) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = data_location.glob('*.csv')
    labels = pd.read_csv(label_location)  # ObjectID,TimeIndex,Direction,Node,Type
    float_size = 'float32'
    datatypes = {"Timestamp": "str", "Eccentricity": float_size, "Semimajor Axis (m)": float_size,
                 "Inclination (deg)": float_size, "RAAN (deg)": float_size,
                 "Argument of Periapsis (deg)": float_size,
                 "True Anomaly (deg)": float_size, "Latitude (deg)": float_size, "Longitude (deg)": float_size,
                 "Altitude (m)": float_size, "X (m)": float_size, "Y (m)": float_size, "Z (m)": float_size,
                 "Vx (m/s)": float_size, "Vy (m/s)": float_size, "Vz (m/s)": float_size}

    loaded_objectIDs = []
    loaded_dfs = []

    # Load out_df
    for i, data_file in enumerate(data_path):
        if i == amount:
            break

        data_df = pd.read_csv(data_file, dtype=datatypes)
        object_id = int(data_file.stem)  # csv is named after its objectID/other way round
        data_df['ObjectID'] = object_id
        data_df['TimeIndex'] = range(len(data_df))
        # convert timestamp from str to float
        data_df['Timestamp'] = (pd.to_datetime(data_df['Timestamp'])).apply(lambda x: x.timestamp()).astype(
            float_size)

        loaded_dfs.append(data_df)
        # append to later just drop all tgts that habe not been loaded
        loaded_objectIDs.append(object_id)

    out_df = pd.concat(loaded_dfs, axis=0)

    out_df_index = pd.MultiIndex.from_frame(out_df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    out_df.index = out_df_index
    # out_df.drop(labels=['ObjectID', 'TimeIndex'], axis=1, inplace=True)
    out_df.sort_index(inplace=True)

    # drop end of study targets as we do not have to predict those
    labels = labels[labels['Type'] != 'ES']
    # if we only load a few csv's, then just load the targets of the loaded objectIDs
    labels = labels[labels['ObjectID'].isin(loaded_objectIDs)]
    labels.reset_index(drop=True, inplace=True)

    return out_df, labels


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

    out_df = pd.DataFrame()  # all data
    labels = pd.read_csv(label_location)  # ObjectID,TimeIndex,Direction,Node,Type

    loaded_dfs = []
    # Load out_df
    for i, data_file in enumerate(data_path):
        if i == amount:
            break
        # these are broken
        if int(data_file.stem) in [30, 113, 1012, 1383, 1385, 1386, 1471, 1473, 1474]:
            continue

        data_df = pd.read_csv(data_file, dtype=datatypes)
        data_df['ObjectID'] = int(data_file.stem)  # csv is named after its objectID/other way round
        data_df['TimeIndex'] = range(len(data_df))
        # convert timestamp from str to float
        data_df['Timestamp'] = (pd.to_datetime(data_df['Timestamp'])).apply(lambda x: x.timestamp()).astype(
            float_size)
        data_df.drop(labels='Timestamp', axis=1, inplace=True)

        # Add EW and NS nodes to data. They are extracted from the labels and converted to integers

        ground_truth_object = labels[labels['ObjectID'] == data_df['ObjectID'][0]].copy()
        # Separate the 'EW' and 'NS' types in the ground truth
        ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
        ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()

        # Create 'EW' and 'NS' labels and fill 'unknown' values
        ground_truth_EW['EW'] = 1.0
        ground_truth_NS['NS'] = 1.0

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

        merged_df["EW_org"] = merged_df["EW"]
        merged_df["NS_org"] = merged_df["NS"]

        indices = merged_df[merged_df['EW'] == 1].index
        seq_len = len(data_df)
        for idx in indices[1:]:
            puffer = 2
            start = idx - puffer if idx - puffer >= 0 else 0
            end = idx + puffer if idx + puffer < seq_len else seq_len
            # debug assert to see if window is smaller
            assert end - start == 4  # 4 not 5 because pandas indexing
            merged_df.loc[start:end, "EW"] = 1

        indices = merged_df[merged_df["NS"] == 1].index
        seq_len = len(data_df)
        for idx in indices[1:]:
            puffer = 2
            start = idx - puffer if idx - puffer >= 0 else 0
            end = idx + puffer if idx + puffer < seq_len else seq_len
            # debug assert to see if window is smaller
            assert end - start == 4  # 4 not 5 because pandas indexing
            merged_df.loc[start:end, "NS"] = 1

        # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
        merged_df['EW'].fillna(0.0, inplace=True)
        merged_df['NS'].fillna(0.0, inplace=True)

        loaded_dfs.append(merged_df)

    out_df = pd.concat(loaded_dfs, axis=0)
    out_df = out_df.astype({"EW": "float32", "NS": "float32"})

    out_df_index = pd.MultiIndex.from_frame(out_df[['ObjectID', 'TimeIndex']], names=['ObjectID', 'TimeIndex'])
    out_df.index = out_df_index
    # out_df.drop(labels=['ObjectID', 'TimeIndex', 'EW', 'NS'], axis=1, inplace=True)
    out_df.sort_index(inplace=True)

    return out_df


def load_first_sample(data_location: Path, label_location: Path, amount: int = -1) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
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

    out_df = pd.DataFrame()  # all data
    labels = pd.read_csv(label_location)  # ObjectID,TimeIndex,Direction,Node,Type
    labels = labels.loc[labels.loc[:, "TimeIndex"] == 0, :]

    loaded_objectIDs = []
    loaded_dfs = []

    # Load out_df
    for i, data_file in enumerate(data_path):
        if i == amount:
            break

        data_df = pd.read_csv(data_file, dtype=datatypes, nrows=1)
        object_id = int(data_file.stem)  # csv is named after its objectID/other way round
        data_df['ObjectID'] = object_id
        data_df['TimeIndex'] = range(len(data_df))
        # convert timestamp from str to float
        data_df['Timestamp'] = (pd.to_datetime(data_df['Timestamp'])).apply(lambda x: x.timestamp()).astype(
            float_size)
        data_df.drop(labels='Timestamp', axis=1, inplace=True)

        loaded_dfs.append(data_df)
        # append to later just drop all tgts that habe not been loaded
        loaded_objectIDs.append(object_id)

    out_df = pd.concat(loaded_dfs, axis=0)

    labels = labels.loc[labels.loc[:, 'ObjectID'].isin(loaded_objectIDs), :]
    # just get the EW and NS direction Series, explicit copy, because pandas is crying
    EW_labels = labels.loc[labels.loc[:, 'Direction'] == 'EW', :].copy()
    NS_labels = labels.loc[labels.loc[:, 'Direction'] == 'NS', :].copy()
    # reset indices for merge into single Series
    EW_labels.reset_index(drop=True, inplace=True)
    NS_labels.reset_index(drop=True, inplace=True)

    # merge into single series. Just use the existing EW Series
    EW_labels.loc[:, "EW"] = EW_labels.loc[:, "Type"]
    EW_labels.loc[:, "NS"] = NS_labels.loc[:, "Type"]
    EW_labels.drop(labels=["Direction", "Node", "Type", "TimeIndex"], axis=1, inplace=True)

    out_df.sort_values(by=['ObjectID'])
    EW_labels.sort_values(by=['ObjectID'])
    out_df.reset_index(drop=True, inplace=True)
    EW_labels.reset_index(drop=True, inplace=True)

    return out_df, EW_labels


def split_train_test(data: pd.DataFrame, train_test_ration: float = 0.8, random_state: int = 42) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # get unique ObjectIDs
    first_level_values: pd.Series = data.index.get_level_values(0).unique().to_series()
    train_indices = first_level_values.sample(frac=train_test_ration, random_state=random_state)
    test_indices = first_level_values.drop(train_indices.index)

    train = data.loc[train_indices]
    test = data.loc[test_indices]

    return train, test


def pad_sequence_vec(src_batch: List[torch.Tensor], padding_vec: torch.Tensor) -> torch.Tensor:
    """
    Equivalent to torch.nn.utils.rnn.pad_sequence but with a vector for the source sequence
    Args:
        src_batch: List of source tensors
        padding_vec: tensor of padding vector
    Returns: torch.Tensor with all sequences. Shape(Batch size, Sequences length (with padded), features size)
    """

    def insert_padding(seq: torch.Tensor):
        num_padding = max_len - len(seq)
        if num_padding == 0:
            return seq.unsqueeze(0)
        # else add padding
        padding_tensor = padding_vec.repeat((num_padding, 1))
        return torch.cat((seq, padding_tensor), dim=0).unsqueeze(0)

    batch_size = len(src_batch)
    length = [len(x) for x in src_batch]
    max_len = max(length)
    min_len = min(length)

    if max_len != min_len:
        # do some padding
        src_batch = torch.cat(list(map(insert_padding, src_batch)), dim=0)
    else:
        src_batch = src_batch[0].repeat((batch_size, 1, 1))

    return src_batch


def convert_tgts_for_eval(pred: torch.Tensor, tgt: torch.Tensor, objectIDs: torch.Tensor,
                          tgt_dict: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    converts tgt and prediction to dataframes containing only the changes as strings.
    Args:
        pred: Tensor, shape ''[batch_size, sequence_len, tgt_feature_size]
        tgt: Tensor, shape ''[batch_size, sequence_len, 1]
        objectIDs: Tensor 1D
        tgt_dict: dict from int to str

    Returns: Two pd.Dataframes containing changes in the sequence in str format

    """
    # copy input and get on cpu
    tgt = tgt.numpy(force=True)  # 'most likely' a copy, forced of the GPU to cpu
    pred = pred.numpy(force=True)
    objectIDs = objectIDs.numpy(force=True)

    tgt = tgt.squeeze(-1)
    # pred = pred.squeeze(-1)
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

        # iterate over sequence and translate each entry from int to str
        time_index: int = 0
        for single_pred, single_tgt in zip(batch_pred, batch_tgt):
            # stop when we have padding,
            if single_tgt == 'PADDING':
                break
            if single_pred == 'PADDING':
                # This (hopefully) never happens.
                raise ValueError("The model predicted PADDING even thought it should not. "
                                 "This is a problem. Try to retrain?")

            EW_pred, NS_pred = single_pred.split('/')
            EW_tgt, NS_tgt = single_tgt.split('/')
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


def state_change_eval(pred: torch.Tensor, tgt: torch.Tensor):
    pred = pred.squeeze()
    tgt = tgt.squeeze()
    seq_len = pred.size(dim=0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(seq_len):
        if pred[i] == tgt[i] == 1:
            TP += 1
        elif pred[i] == 1 and tgt[i] == 0:
            FP += 1
        elif pred[i] == 0 and tgt[i] == 1:
            FN += 1
        elif pred[i] == tgt[i] == 0:
            TN += 1

    return TP, FP, TN, FN


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
