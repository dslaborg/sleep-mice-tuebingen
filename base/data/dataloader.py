import logging

import numpy as np
import tables
import torch.utils.data as tud

from base.config_loader import ConfigLoader
from base.data.data_augmentor import DataAugmentor
from base.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL
from base.utilities import format_dictionary

logger = logging.getLogger('tuebingen')


class TuebingenDataloader(tud.Dataset):
    """ dataloader for data from Tuebingen, requires the data to be stored in a pytables table with the structure
    described in `data_table.py`

    Each row in the table contains the data and label for one sample, without SAMPLES_LEFT and SAMPLES_RIGHT. This
    means, that every sample can be identified by it's index in the table, which is used during rebalancing. """

    def __init__(self, config, dataset, balanced=False, augment_data=True, data_fraction=1.0):
        """
        Args:
             config (ConfigLoader): config of the running experiment
             dataset (str): dataset from which to load data, must be a table in the pytables file
             balanced (bool): flag, whether the loaded data should be rebalanced by using BALANCING_WEIGHTS
             augment_data (bool): flag, whether the data is to be augmented, see `DataAugmentor`
        """
        self.config = config
        self.augment_data = augment_data
        self.dataset = dataset
        self.balanced = balanced
        self.data_fraction = data_fraction
        self.data = None
        self.data_augmentor = DataAugmentor(config)
        self.max_idx = 0

        # file has to be opened here, so the indices for each stage can be loaded
        self.file = tables.open_file(self.config.DATA_FILE)
        self.stages = self.get_stage_data()
        # max index is needed to calculate limits for the additional samples loaded by SAMPLES_LEFT and SAMPLES_RIGHT
        self.nitems = sum([len(s) for s in self.stages])
        self.indices = self.get_indices()
        self.file.close()

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        if self.data is None:  # open in thread
            self.file = tables.open_file(self.config.DATA_FILE)
            self.data = self.file.root[self.dataset]

        # load internal index for rebalancing purposes, see get_indices()
        index = self.indices[index]

        # load one additional sample to each side for window warping and time shift
        left = self.config.SAMPLES_LEFT + 1
        right = self.config.SAMPLES_RIGHT + 1
        # calculate start and end to prevent IndexErrors
        idx_from = 0 if index - left < 0 else index - left
        idx_to = idx_from + left + right
        if idx_to >= self.max_idx:
            idx_from = self.max_idx - left - right - 1
            idx_to = idx_from + left + right
        index = int((idx_from + idx_to) / 2)

        # if the samples in the block are not from the same mouse, the block has to be shifted
        if not np.all(self.data[index][COLUMN_MOUSE_ID] == self.data[idx_from:idx_to + 1][COLUMN_MOUSE_ID]):
            idx = np.where(self.data[index][COLUMN_MOUSE_ID] != self.data[idx_from:idx_to + 1][COLUMN_MOUSE_ID])[0][0]
            dist_from_limits = min(idx, idx_to - idx_from - idx)  # how much to shift
            if dist_from_limits == idx:  # sample from wrong mouse is on the left side --> shift to the right
                idx_shift = dist_from_limits
            else:  # sample from wrong mouse is on the right side --> shift to the left
                idx_shift = -dist_from_limits
            idx_from += idx_shift
            idx_to += idx_shift
            index += idx_shift

        # load only the data specified by SAMPLES_LEFT and SAMPLES_RIGHT w/o the samples for window warping
        rows = self.data[idx_from + 1:idx_to]
        feature = np.c_[[rows[c].flatten() for c in self.config.CHANNELS]]

        # load samples to the left and right and use them for data augmentation
        sample_left = np.c_[[self.data[idx_from][c].flatten() for c in self.config.CHANNELS]]
        sample_right = np.c_[[self.data[idx_to][c].flatten() for c in self.config.CHANNELS]]
        if self.augment_data:
            feature = self.data_augmentor.alternate_signals(feature, sample_left, sample_right)

        # transform the label to it's index in STAGES
        return feature, self.config.STAGES.index(str(self.data[index][COLUMN_LABEL], 'utf-8'))

    def __len__(self):
        return self.nitems

    def get_indices(self):
        """ loads indices of samples in the pytables table the dataloader returns

        if flag `balanced` is set, rebalancing is done here by randomly drawing samples from all samples in a stage
        until nitems * BALANCING_WEIGHTS[stage] is reached

        drawing of the samples is done with replacement, so samples can occur more than once in the dataloader """
        indices = np.empty(0)

        data_dist = {s: len(n) for s, n in zip(self.config.STAGES, self.stages)}
        logger.info(
            'data distribution in database for dataset ' + str(self.dataset) + ':\n' + format_dictionary(data_dist))

        # apply balancing
        if self.balanced:
            # the balancing weights are normed
            # if a stage has no samples, the weight belonging to it is set to 0
            balancing_weights = np.array(self.config.BALANCING_WEIGHTS, dtype='float')
            for n, stage_data in enumerate(self.stages):
                if len(stage_data) == 0:
                    print('label ' + self.config.STAGES[n] + ' has zero samples')
                    balancing_weights[n] = 0
            balancing_weights /= sum(balancing_weights)

            # draw samples according to balancing weights
            for n, z in enumerate(zip(self.stages, self.config.STAGES)):
                stage_data, stage = z
                if len(stage_data) == 0:
                    continue
                indices = np.r_[indices, np.random.choice(stage_data, size=int(
                    self.nitems * balancing_weights[n]) + 1, replace=True)].astype('int')
                data_dist[stage] = int(self.nitems * balancing_weights[n]) + 1
            np.random.shuffle(indices)  # shuffle indices, otherwise they would be ordered by stage...
        else:  # if 'balanced' is not set, all samples are loaded
            for stage_data in self.stages:
                indices = np.r_[indices, stage_data].astype('int')
            indices = np.sort(indices)  # the samples are sorted by index for the creation of a transformation matrix

        logger.info('data distribution after processing:\n' + format_dictionary(data_dist))

        return indices

    def reset_indices(self):
        """ reload indices, only relevant for balancing purposes, because the samples are redrawn """
        self.indices = self.get_indices()

    def get_stage_data(self):
        """ load indices of samples in the pytables table for each stage

        if data_fraction is set, load only a random fraction of the indices

        Returns:
            list: list with entries for each stage containing lists with indices of samples in that stage
        """
        stages = []
        table = self.file.root[self.dataset]

        for stage in self.config.STAGES:
            stages.append(table.get_where_list('({}=="{}")'.format(COLUMN_LABEL, stage)))
        self.max_idx = np.max(stages)

        if self.config.DATA_FRACTION_STRAT is None or self.dataset != 'train':
            for i, stage_data in enumerate(stages):
                np.random.shuffle(stage_data)
                stages[i] = stage_data[:int(self.data_fraction * len(stage_data))]
        else:
            if self.config.DATA_FRACTION_STRAT == 'uniform':
                num_samples = int(self.data_fraction * sum([len(s) for s in stages]) / len(self.config.STAGES))
                num_samples = min(num_samples, min([len(s) for s in stages]))
                stages = [s[:num_samples] for s in stages]

        return stages
