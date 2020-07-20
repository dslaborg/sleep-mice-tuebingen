#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, default='exp001')
opt = parser.parse_args()

from sys import path
path.insert(0, '.')

import keras
from os.path import join

import tensorflowjs as tfjs

from base.config_loader import ConfigLoader

def version(module):
    return f"{module.__name__}" + (f" v{module.__version__}" \
           if hasattr(module, "__version__") \
           else "")

print(version(keras))
print(version(tfjs))
assert keras.__version__ == '2.2.2', version(keras)
assert tfjs.__version__ == '0.8.0', version(tfjs)

def build_keras_model(config):
    filename = join(config.EXPERIMENT_DIR, 'keras', 'model.h5')
    return keras.models.load_model(filename)

config = ConfigLoader(opt.experiment)
keras_model = build_keras_model(config) 
keras_model.summary()
filename = join(config.EXPERIMENT_DIR, 'tfjs')
tfjs.converters.save_keras_model(keras_model, filename)
