# Tuebingen

Repo containing the code to reproduce sleep classification results on data 
from tuebingen.

## requirements 
The experiments where run in an anaconda environment on python 3.7 that can be 
recreated using requirements_win.txt for windows or requirements_linux.txt for
linux.


## scripts
### utils
#### data_viewer.py
script to visualize the data after transformation and predictions of the best model
in the selected experiment on this data  
**Parameters:** 
* -e, --experiment: name of experiment to load config from, there must be a file
\<experiment\>.py in the config folder
* -d, --dataset: dataset to load data from

#### group_samples_by_stage.py
counts samples per stage and dataset in the transformed `DATA_FILE`  
**Parameters:**
* -e, --experiment: name of experiment to load config from, there must be a file
\<experiment\>.py in the config folder

### general
#### run_experiment.py
trains the model specified in the experiment using the configured parameters and 
evaluates it, only the model from the epoch with best validation f1-score is saved  
**Parameters:**
* -e, --experiment: name of experiment to run, there must be a file
\<experiment\>.py in the config folder

#### transform_files_tuebingen.py
script to transform data from the given h5 format into a pytables table, also 
performs preprocessing steps like downsampling  
**Parameters:**
* -e, --experiment: name of experiment to transform data to, there must be a file
\<experiment\>.py in the config folder



## configuration parameters
short description of the possible configuration parameters and their mapping to 
fields in ConfigLoader  
format: \<parameter name in .yml\> (`<mapped name in ConfigLoader>`)
* general
    * device (`DEVICE`): where net is trained and evaluated, can be either 'gpu' 
    or 'cpu'
* dirs
    * cache (`-`): directory used as a cache for various files, like the transformed 
    data file
    * data (`DATA_DIR`): directory on your pc containing original data files
* data
    * sample_duration (`SAMPLE_DURATION`): duration of a sample in seconds
    * sampling_rate (`SAMPLING_RATE`): sampling rate of the signal in Hz
    * scoring_map (`SCORING_MAP`): used during data transformation to map
    scores in h5 files to the known stages; format/example:
        * \<stage to map to\>: \<list of scores to map\>
        * Wake: [1, 17]
    * stage_map (`STAGE_MAP`): describes if stages should be mapped to another 
    stage, useful for omitting intermediate stages like 'Pre REM'; format/example:
        * \<stage to map\>: \<stage to map to\>
        * Pre REM: 'REM'
* experiment
    * data
        * split (`DATA_SPLIT`): map containing a key for each dataset with a list
        of mice; used during data transformation to map h5 files to the
        corresponding dataset in the pytables table; there must at least exist
        entries for datasets `train` and `valid`
        * file (`DATA_FILE`): name of the file the transformed pytables table 
        is saved in; file is created in `dirs.cache`
        * stages (`STAGES`): list of stages you want to train your model on and
        predict
        * balancing_weights (`BALANCING_WEIGHTS`): list of weights used for 
        rebalancing, see data_table.py for details
        * channels (`CHANNELS`): list of channels in your data files, that are 
        used as features
        * samples_left (`SAMPLES_LEFT`): number of additional samples to the left
        that are loaded together with the sample to classify; the input of the net 
        consists of `samples_left`, the sample to classify and `samples_right`
        * samples_right (`SAMPLES_RIGHT`): number of additional samples to the 
        right, for further information see `samples_left`
    * training
        * log_interval (`LOG_INTERVAL`): percentage of data in trainloader after
        which a log message is created in an epoch
        * additional_model_safe (`EXTRA_SAFE_MODELS`): if set to `True` additional
        snapshots of the best model of a run are saved in `MODELS_DIR`
        * batch_size (`BATCH_SIZE`): batch size for the train dataloader
        * data_fraction (`DATA_FRACTION`): fraction of train data to use for 
        training, validation data remains the same
        * epochs (`EPOCHS`): number of epochs to train
        * optimizer
            * scheduler: see scheduled_optim.py for details
                * warmup_epochs (`WARMUP_EPOCHS`): number of warmup epochs
                * mode (`S_OPTIM_MODE`): mode used for learning rate decrease 
                after warmup
                * parameters (`S_OPTIM_PARAS`): list of parameters for the 
                selected mode
            * learning_rate (`LEARNING_RATE`): peak learning rate after warmup
            * class (`OPTIMIZER`): subclass of `torch.optim.optimizer.Optimizer`
            * parameters (`OPTIM_PARAS`): dict of parameters for the `OPTIMIZER` 
            like `eps`, `betas`, etc. 
            * l1_weight_decay (`L1_WEIGHT_DECAY`): factor of applied L1 decay
            * l2_weight_decay (`L2_WEIGHT_DECAY`): factor of applied L2 decay

    * evaluation
        * batch_size (`BATCH_SIZE_EVAL`): batch size used during evaluation

    * model
        * filters (`FILTERS`): number of filters used in conv layers of the 
        feature extractor
        * classifier_dropout (`CLASSIFIER_DROPOUT`): list of dropout probabilities
        applied in the classifier
        * feature_extr_dropout (`FEATURE_EXTR_DROPOUT`): list of dropout 
        probabilities applied in the feature extractor
        * name (`MODEL_NAME`): name of the model to be trained, must be the name of
        a python file in `base.models`; the name of the class in the file must be 
        'Model'

    * data_augmentation: for more details on the types of data augmentation see
    data_augmentor.py
        * gain (`GAIN`): parameter for signal amplitude amplification
        * flip (`FLIP`): probability for vertically flipping single datapoints 
        in the signal
        * flip_all (`FLIP_ALL`): probability for vertically flipping the whole 
        signal
        * flip_hori (`FLIP_HORI`): probability for horizontally flipping the
        whole signal
        * window_warp_size (`WINDOW_WARP_SIZE`): factor for selection of the 
        new window size in window warping
        * time_shift (`TIME_SHIFT`): factor to determine the random amount of time
        a signal is shifted