# Tuebingen

Repository containing the code to reproduce sleep classification results on data 
from tuebingen.

<br>

## Install Requirements

The experiments where run in an anaconda environment on python 3.7 that can be
recreated using *requirements_win.txt* for windows or *requirements_linux.txt*
for linux.  Since it does not seem like conda can install pytorch from a
requirements file it must be installed manually. See
https://pytorch.org/get-started/locally/ for detailed instructions.

### Linux

Run:
```bash
conda create --name mice --file requirements_linux.txt
conda activate mice
pip install torch
```

## Prepare Dataset for ML

###

Raw electrophysiological signals from each epoch are stored in a pytables
object.  For a mouse recording consisting of several consecutive scored epoch,
we store in columns the mouse id, signal by channel, and the assigned stage
label.  See [our Tuebingen-data script](scripts/transform_files_tuebingen.py)
for an example.  The length of epoch segments, sampling rates, and labels have
to be in line with the experiment configuration file such as [this default
one](./config/standard_config.yml).

## scripts
#### data_viewer.py
script to visualize the data in *dataset* after transformation and to show 
predicted probabilities of the best model of the passed *experiment*

**Parameters:** 
* -e, --experiment: name of experiment to load config from, there must be a file
*\<experiment\>.py* in the config folder
* -d, --dataset: dataset to load data from

#### group_samples_by_stage.py
counts samples per stage and dataset in the transformed `DATA_FILE`

**Parameters:**
* -e, --experiment: name of experiment to load config from, there must be a file
*\<experiment\>.py* in the config folder

#### run_experiment.py
trains the model specified in the *experiment* using the configured parameters and 
evaluates it; only the model from the epoch with best validation f1-score is saved

**Parameters:**
* -e, --experiment: name of experiment to run, there must be a file
*\<experiment\>.py* in the config folder

#### transform_files_tuebingen.py
script to transform data from the given h5 format into a pytables table, also 
performs preprocessing steps like downsampling

**Parameters:**
* -e, --experiment: name of experiment to transform data to, there must be a file
*\<experiment\>.py* in the config folder

### evaluate_experiment.py
evaluates best model in given *experiment* on data from passed dataset; results are 
loggedin a log file in `EXPERIMENT_DIR` and plots are created in `VISUALS_DIR`

**Parameters:**
* -e, --experiment: name of experiment to evaluate best model from, there must 
be a file *\<experiment\>.py* in the config folder
* -d, --dataset: dataset to load data from, there must exist a corresponding table
in your transformed data 

### Torch to tensorflow.js

#### tfjs/torch\_to\_keras.py

Transform torch model to keras model.

**Parameters**

* -e, --experiment: name of experiment to evaluate best model from, there must
  be a file *\<experiment\>.py* in the config folder

#### tfjs/torch\_vs\_keras.py

Compare torch model with transformed keras model.

**Parameters**

* -e, --experiment: name of experiment to evaluate best model from, there must
  be a file *\<experiment\>.py* in the config folder

<br>

## configuration parameters
There are many parameters in the code that can be configured in an external yaml
file. The configuration files must be in the folder **/config**.  
Currently there already are two files in the directory, the *standard_config.yml*
describing the standard configuration for the data, training, model, etc and an 
example config file *exp001.py* describing the example leading to the best results 
I found. As you can see the configuration done in *exp001* is very little compared
to the *standard_config*. This is because all additional config files complement
the *standard_config*, i.e. parameters missing in *exp001* are instead loaded
from *standard_config*.

The following list shows a short description of the possible configuration 
parameters and their mapping to the fields in *ConfigLoader.py*. For examples 
of the configurations see *standard_config.yml*.  
format: \<parameter name in .yml\> (`<mapped name in ConfigLoader>: <dtype>`)

**Info:** only the value `null` in a yaml file is converted to `None` in the code, 
`None` gets converted to a string `'None'`

* general
    * device (`DEVICE: ['cpu' or 'gpu']`): where net is trained and evaluated, 
    can be either 'gpu' or 'cpu'
* dirs
    * cache (`-`): path to a directory used as a cache for various files, like 
    the transformed data file
    * data (`DATA_DIR: str`): path to a directory on your pc containing the original
    data files
* data
    * sample_duration (`SAMPLE_DURATION: [int or float]`): duration of a sample 
    in seconds
    * sampling_rate (`SAMPLING_RATE: int`): sampling rate of the signal in Hz
    * scoring_map (`SCORING_MAP: dict`): used during data transformation to map
    scores in h5 files to the known stages; format/example:
        * \<stage to map to\>: \<list of scores to map\>
        * Wake: [1, 17]
    * stage_map (`STAGE_MAP: dict`): describes if stages should be mapped to another 
    stage, useful for omitting intermediate stages like 'Pre REM'; format/example:
        * \<stage to map\>: \<stage to map to\>
        * Pre REM: 'REM'
* experiment
    * data
        * split (`DATA_SPLIT: dict`): map containing a key for each dataset with a 
        list of mice; used during data transformation to map h5 files to the
        corresponding dataset in the pytables table; there must at least exist
        entries for datasets `train` and `valid`
        * file (`DATA_FILE: str`): name of the file the transformed data 
        is saved in; file is created in `dirs.cache`
        * stages (`STAGES: dict`): list of stages you want to train your model on 
        and predict
        * balanced_training (`BALANCED_TRAINING: bool`): decides if rebalancing 
        is applied during training (`True`: rebalancing, `False`: no rebalancing)
        * balancing_weights (`BALANCING_WEIGHTS: list[float]`): list of weights 
        used for rebalancing, see *data_table.py* for details
        * channels (`CHANNELS: list[str]`): list of channels in your data files, 
        that are used as features
        * samples_left (`SAMPLES_LEFT: int`): number of additional samples to the
        left that are loaded together with the sample to classify; the input of 
        the net consists of `samples_left`, the sample to classify and `samples_right`
        * samples_right (`SAMPLES_RIGHT: int`): number of additional samples to the 
        right, for further information see `samples_left`
    * training
        * log_interval (`LOG_INTERVAL: int`): percentage of data in trainloader 
        after which a log message is created in an epoch
        * additional_model_safe (`EXTRA_SAFE_MODELS: bool`): if set to `True` 
        additional snapshots of the best model of a run are saved in `MODELS_DIR`
        * batch_size (`BATCH_SIZE: int`): batch size for the train dataloader
        * data_fraction (`DATA_FRACTION: float`): fraction of train data to use for 
        training, validation data remains the same
        * data_fraction_strat (`DATA_FRACTION_STRAT: ['uniform', null]`): strategy 
        for data fractions, currently supports only null (no strategy, take the 
        same data fraction of each stage) and 'uniform' (take the same number of 
        samples from each stage so the total number of samples corresponds to the
        data fraction)
        * epochs (`EPOCHS: int`): number of epochs to train
        * optimizer
            * scheduler: for details and examples see *scheduled_optim.py*
                * warmup_epochs (`WARMUP_EPOCHS: int`): number of warmup epochs
                * mode (`S_OPTIM_MODE: ['exp', 'plat', 'step', 'half', null]`): 
                mode used for learning rate decrease after warmup
                * parameters (`S_OPTIM_PARAS: dict`): list of parameters for the 
                selected mode
            * learning_rate (`LEARNING_RATE: float`): peak learning rate after 
            warmup
            * class (`OPTIMIZER: Optimizer`): subclass of 
            `torch.optim.optimizer.Optimizer`
            * parameters (`OPTIM_PARAS: dict`): parameters for the `OPTIMIZER` 
            like `eps`, `betas`, etc. 
            * l1_weight_decay (`L1_WEIGHT_DECAY: float`): factor of applied L1 decay
            * l2_weight_decay (`L2_WEIGHT_DECAY: float`): factor of applied L2 decay

    * evaluation
        * batch_size (`BATCH_SIZE_EVAL: int`): batch size used during evaluation

    * model
        * filters (`FILTERS: int`): number of filters used in conv layers of the 
        feature extractor
        * classifier_dropout (`CLASSIFIER_DROPOUT: list[float]`): list of dropout 
        probabilities applied in the classifier
        * feature_extr_dropout (`FEATURE_EXTR_DROPOUT: list[float]`): list of 
        dropout probabilities applied in the feature extractor
        * name (`MODEL_NAME: str`): name of the model to be trained, must be the 
        name of a python file in `base.models`; the name of the class in the file
        must be 'Model'

    * data_augmentation: for more details on the types of data augmentation see
    *data_augmentor.py*
        * gain (`GAIN: float`): parameter for signal amplitude amplification
        * flip (`FLIP: float`): probability for vertically flipping single 
        datapoints in the signal
        * flip_all (`FLIP_ALL: float`): probability for vertically flipping the 
        whole signal
        * flip_hori (`FLIP_HORI: float`): probability for horizontally flipping the
        whole signal
        * window_warp_size (`WINDOW_WARP_SIZE: float`): factor for selection of the 
        new window size in window warping
        * time_shift (`TIME_SHIFT: float`): factor to determine the random amount 
        of time a signal is shifted
