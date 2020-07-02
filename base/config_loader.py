import collections
import time
from os import makedirs
from os.path import dirname, join, realpath

import numpy as np
import yaml
from torch import optim


def update_dict(d_to_update: dict, update: dict):
    """method to update a dict with the entries of another dict

    `d_to_update` is updated by `update`"""
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            d_to_update[k] = update_dict(d_to_update.get(k, {}), v)
        else:
            d_to_update[k] = v
    return d_to_update


class ConfigLoader:
    def __init__(self, experiment='standard_config'):
        """general class to load config from yaml files into fields of an instance of this class

        first the config from standard_config.yml is loaded and then updated with the entries in the config file
        specified by `experiment`

        for a description of the various configurations see README.md

        Args:
            experiment (str): name of the config file to load without the .yml file extension; file must be in folder
                'config'
        """
        base_dir = realpath(join(dirname(__file__), '..'))

        self.experiment = experiment
        config = self.load_config()

        self.RUN_NAME = experiment + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        """identifier for the run of the experiment, used in log_file and `VISUALS_DIR`"""

        # general
        self.DEVICE = config['general']['device']
        assert self.DEVICE in ['cpu', 'cuda'], 'DEVICE only support `cpu` or `cuda`'

        # dirs
        self.EXPERIMENT_DIR = join(base_dir, 'results', self.experiment)
        """general directory with results from experiment"""
        makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        self.MODELS_DIR = join(self.EXPERIMENT_DIR, 'models')
        """directory in `EXPERIMENT_DIR` for `extra_safe` models"""
        makedirs(self.MODELS_DIR, exist_ok=True)
        self.VISUALS_DIR = join(self.EXPERIMENT_DIR, 'visuals', self.RUN_NAME)
        """directory in `EXPERIMENT_DIR` for all generated plots"""
        makedirs(self.VISUALS_DIR, exist_ok=True)

        self.DATA_DIR = realpath(config['dirs']['data'])
        makedirs(self.DATA_DIR, exist_ok=True)
        cache_dir = config['dirs']['cache']
        makedirs(cache_dir, exist_ok=True)

        # data
        self.SAMPLE_DURATION = config['data']['sample_duration']
        self.SAMPLING_RATE = config['data']['sampling_rate']
        self.SCORING_MAP = config['data']['scoring_map']
        self.STAGE_MAP = config['data']['stage_map']

        # experiment
        exp_config = config['experiment']
        # data
        data_config = exp_config['data']
        self.DATA_SPLIT = data_config['split']
        self.DATA_FILE = join(cache_dir, data_config['file'])
        self.STAGES = data_config['stages']
        for k in self.STAGE_MAP:
            msg = 'You are trying to map STAGE {} to STAGE {} that does not exist in experiment.data.stages'
            assert self.STAGE_MAP[k] in self.STAGES, msg.format(k, self.STAGE_MAP[k])
        for k in self.SCORING_MAP:
            msg = 'You are trying to map scores to STAGE {} that does not exist in experiment.data.stages'
            assert k in self.STAGES, msg.format(k)
        for s in self.STAGES:
            assert s in self.STAGE_MAP.values(), 'STAGE {} has no mapping in STAGE_MAP'.format(s)
        self.BALANCING_WEIGHTS = data_config['balancing_weights']
        assert np.all([w >= 0 for w in self.BALANCING_WEIGHTS]), 'BALANCING_WEIGHTS must be non negative'
        self.CHANNELS = data_config['channels']
        self.SAMPLES_LEFT = data_config['samples_left']
        self.SAMPLES_RIGHT = data_config['samples_right']

        # training
        training_config = exp_config['training']
        self.LOG_INTERVAL = training_config['log_interval']
        self.EXTRA_SAFE_MODELS = training_config['additional_model_safe']
        self.BATCH_SIZE = training_config['batch_size']
        self.DATA_FRACTION = training_config['data_fraction']
        self.DATA_FRACTION_STRAT = training_config['data_fraction_strat']
        assert self.DATA_FRACTION_STRAT in ['uniform', None], \
            'currently only "uniform" or None is supported as a data fraction strategy'
        self.EPOCHS = training_config['epochs']
        self.WARMUP_EPOCHS = training_config['optimizer']['scheduler']['warmup_epochs']
        self.S_OPTIM_MODE = training_config['optimizer']['scheduler']['mode']
        assert self.S_OPTIM_MODE in ['step', 'exp', 'half', 'plat', None], \
            'S_OPTIM_MODE is not one of ["step", "exp", "half", "plat", None]'
        self.S_OPTIM_PARAS = training_config['optimizer']['scheduler']['parameters']
        assert type(self.S_OPTIM_PARAS) in [list, type(None)], 'S_OPTIM_PARAS must be a list or None'
        self.LEARNING_RATE = training_config['optimizer']['learning_rate']
        self.OPTIMIZER = getattr(optim, training_config['optimizer']['class'])
        assert issubclass(self.OPTIMIZER, optim.Optimizer), 'OPTIMIZER must be a subtype of optim.optimizer.Optimizer'
        self.OPTIM_PARAS = training_config['optimizer']['parameters']
        assert type(self.OPTIM_PARAS) == dict, 'OPTIM_PARAS must be a dict'
        self.L1_WEIGHT_DECAY = training_config['optimizer']['l1_weight_decay']
        self.L2_WEIGHT_DECAY = training_config['optimizer']['l2_weight_decay']

        # evaluation
        self.BATCH_SIZE_EVAL = config['experiment']['evaluation']['batch_size']

        # model
        model_config = exp_config['model']
        self.FILTERS = model_config['filters']
        self.CLASSIFIER_DROPOUT = model_config['classifier_dropout']
        self.FEATURE_EXTR_DROPOUT = model_config['feature_extr_dropout']
        self.MODEL_NAME = model_config['name']

        # data augmentation
        data_aug_config = exp_config['data_augmentation']
        self.GAIN = data_aug_config['gain']
        assert 0 <= self.GAIN <= 1, 'GAIN must be in [0,1]'
        self.FLIP = data_aug_config['flip']
        assert 0 <= self.FLIP <= 1, 'FLIP must be in [0,1]'
        self.FLIP_ALL = data_aug_config['flip_all']
        assert 0 <= self.FLIP_ALL <= 1, 'FLIP_ALL must be in [0,1]'
        self.WINDOW_WARP_SIZE = data_aug_config['window_warp_size']
        assert 0 <= self.WINDOW_WARP_SIZE <= 1, 'WINDOW_WARP_SIZE must be in [0,1]'
        self.FLIP_HORI = data_aug_config['flip_hori']
        assert 0 <= self.FLIP_HORI <= 1, 'FLIP_HORI must be in [0,1]'
        self.TIME_SHIFT = data_aug_config['time_shift']
        assert 0 <= self.TIME_SHIFT <= 1, 'TIME_SHIFT must be in [0,1]'

    def load_config(self):
        """loads config from standard_config.yml and updates it with <experiment>.yml"""
        base_dir = realpath(join(dirname(__file__), '..'))
        with open(join(base_dir, 'config', 'standard_config.yml'), 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)
        with open(join(base_dir, 'config', self.experiment + '.yml'), 'r') as ymlfile:
            config = update_dict(config, yaml.safe_load(ymlfile))
        return config
