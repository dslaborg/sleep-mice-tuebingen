#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, default='exp001')
opt = parser.parse_args()

from sys import path
path.insert(0, '.')

from importlib import import_module
from os import makedirs
from os.path import join, isfile
import keras
import numpy as np
import torch
import tensorflow

from base.config_loader import ConfigLoader
from base.logger import Logger

def version(module):
    return f"{module.__name__}" + (f" v{module.__version__}" \
           if hasattr(module, "__version__") \
           else "")

print(version(torch))
print(version(tensorflow))
print(version(keras))
assert keras.__version__ == '2.2.2', version(keras)

def build_keras_model(config):
    filename = join(config.EXPERIMENT_DIR, 'keras', 'model.h5')
    return keras.models.load_model(filename)

def build_torch_model(config):
    # create empty model from model name in config and set it's state from best model in EXPERIMENT_DIR
    module = import_module('.' + config.MODEL_NAME, 'base.models')
    model = module.Model(config).to(config.DEVICE)
    model = model.eval()
    model_file = join(config.EXPERIMENT_DIR, config.MODEL_NAME + '-best.pth')
    if isfile(model_file):
        state = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
    else:
        raise ValueError('model_file {} does not exist'.format(model_file))
    return model

def torch_input_shape(config):
    num_input_channels = len(config.CHANNELS)
    num_samples = config.SAMPLES_LEFT + 1 + config.SAMPLES_RIGHT
    segment_length = int(num_samples * config.SAMPLE_DURATION * config.SAMPLING_RATE)
    return (1, num_input_channels, segment_length)

config = ConfigLoader(opt.experiment)
batch_size, num_input_channels, segment_length = torch_input_shape(config)

keras_model = build_keras_model(config) 
torch_model = build_torch_model(config) 

numpy_tensor = np.zeros((1, segment_length, len(config.CHANNELS))).astype(np.float32)
torch_tensor = torch.from_numpy(numpy_tensor).transpose(2, 1)

torch_output = torch_model(torch_tensor).detach().numpy() 
output = {
    'torch': torch_output,
    'exp(torch)': np.exp(torch_output),
    'keras': keras_model.predict(numpy_tensor),
}

for lib, result in output.items():
    print(lib, result)
