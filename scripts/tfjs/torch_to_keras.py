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
import torch
import tensorflow
from pytorch2keras import converter
import pytorch2keras
import keras

from base.config_loader import ConfigLoader
from base.logger import Logger

def version(module):
    return f"{module.__name__}" + (f" v{module.__version__}" \
           if hasattr(module, "__version__") \
           else "")

print(version(torch))
print(version(tensorflow))
print(version(keras))
print(version(pytorch2keras))
assert torch.__version__ == '0.4.0', version(torch)
assert tensorflow.__version__ == '1.12.0', version(tensorflow)


def build_torch_model(config):
    module = import_module('.' + config.MODEL_NAME, 'base.models')
    model = module.Model(config).to(config.DEVICE)
    return model.eval()

def load_torch_model(config):
    # create empty model from model name in config and set it's state from best model in EXPERIMENT_DIR
    model = build_torch_model(config)
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

torch_model = load_torch_model(config)

batch_size, num_input_channels, segment_length = torch_input_shape(config)
input_tensor = torch.zeros(batch_size, num_input_channels, segment_length)

keras_model = converter.pytorch_to_keras(
    torch_model, input_tensor, input_shape=(num_input_channels, segment_length),
    verbose=True, change_ordering=True)

keras_model_path = join(config.EXPERIMENT_DIR, 'keras')
makedirs(keras_model_path, exist_ok=True)
filename = join(keras_model_path, 'model.h5')
keras_model.save(filename)
print()
print(f"Keras model ready at '{filename}'")
print()
