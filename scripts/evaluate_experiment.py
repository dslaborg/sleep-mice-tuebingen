import argparse
import sys
from importlib import import_module
from os.path import basename, join, dirname, realpath, isfile

import torch
import torch.utils.data as t_data

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader
from base.logger import Logger
from base.evaluation.evaluate_model import evaluate
from base.evaluation.result_logger import ResultLogger


def parse():
    parser = argparse.ArgumentParser(description='evaluate exp')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to run')
    parser.add_argument('--dataset', '-d', default='valid',
                        help='dataset to evaluate model on')

    return parser.parse_args()


def evaluation():
    """evaluates best model in experiment on given dataset"""
    logger.fancy_log('start evaluation')
    result_logger = ResultLogger(config)

    # create dataloader for given dataset, the data should not be altered in any way
    map_loader = TuebingenDataloader(config, args.dataset, balanced=False, augment_data=False)
    dataloader = t_data.DataLoader(map_loader, batch_size=config.BATCH_SIZE_EVAL, shuffle=False, num_workers=4)

    # create empty model from model name in config and set it's state from best model in EXPERIMENT_DIR
    model = import_module('.' + config.MODEL_NAME, 'base.models').Model(config).to(config.DEVICE).eval()
    model_file = join(config.EXPERIMENT_DIR, config.MODEL_NAME + '-best.pth')
    if isfile(model_file):
        model.load_state_dict(torch.load(model_file)['state_dict'])
    else:
        logger.fancy_log('{} does not exist'.format(model_file))
        return
    logger.logger.info('loaded model:\n' + str(model))

    # evaluate model
    labels, _ = evaluate(config, model, dataloader)

    # log/plot results
    result_logger.log_sleep_stage_f1_scores(labels['actual'], labels['predicted'], args.dataset)
    logger.logger.info('')
    result_logger.log_confusion_matrix(labels['actual'], labels['predicted'], args.dataset, wo_plot=False)
    result_logger.log_transformation_matrix(labels['actual'], labels['predicted'], args.dataset,
                                            wo_plot=False)

    logger.fancy_log('finished evaluation')


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment)

    logger = Logger(config)  # wrapper for logger
    logger.init_log_file(args, basename(__file__))  # create log file and log config, etc

    logger.fancy_log('evaluate best model of experiment {} on dataset {}'.format(args.experiment, args.dataset))
    # perform evaluation
    evaluation()
