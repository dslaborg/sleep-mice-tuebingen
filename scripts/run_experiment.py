import argparse
import sys
import time
from importlib import import_module
from os.path import basename, realpath, join, dirname

import torch.utils.data as t_data
from sklearn.metrics import f1_score

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader
from base.evaluation.evaluate_model import evaluate
from base.evaluation.result_logger import ResultLogger
from base.logger import Logger
from base.training.scheduled_optim import ScheduledOptim
from base.training.train_model import train, snapshot


def parse():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to run')

    return parser.parse_args()


def training():
    result_logger = ResultLogger(config)

    dl_train = TuebingenDataloader(config, 'train', True, augment_data=True, data_fraction=config.DATA_FRACTION)
    dl_valid = TuebingenDataloader(config, 'valid', False, augment_data=False)
    trainloader = t_data.DataLoader(dl_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    validationloader = t_data.DataLoader(dl_valid, batch_size=config.BATCH_SIZE_EVAL, shuffle=False, num_workers=4)

    model = import_module('.' + config.MODEL_NAME, 'base.models').Model(config).to(config.DEVICE)
    logger.logger.info('classifier:\n' + str(model))
    optimizer = ScheduledOptim(
        config.OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()),
                         weight_decay=config.L2_WEIGHT_DECAY, **config.OPTIM_PARAS),
        peak_lr=config.LEARNING_RATE, warmup_epochs=config.WARMUP_EPOCHS, total_epochs=config.EPOCHS,
        parameters=config.S_OPTIM_PARAS, mode=config.S_OPTIM_MODE)

    best_epoch = 0
    best_avg_f1_score = 0
    f1_scores = {'train': {'avg': []},
                 'valid': {stage: [] for stage in config.STAGES + ['avg']}}
    losses = {'train': [], 'valid': []}

    for epoch in range(1, config.EPOCHS + 1):
        start = time.time()
        if epoch > 1:
            dl_train.reset_indices()
        optimizer.inc_epoch()

        labels_train, loss_train = train(config, epoch, model, optimizer, trainloader)
        f1_scores['train']['avg'].append(f1_score(labels_train['actual'], labels_train['predicted'], average='macro'))
        losses['train'].append(loss_train)

        labels_valid, loss_valid = evaluate(config, model, validationloader)
        losses['valid'].append(loss_valid)

        logger.logger.info('')
        f1_scores_valid = result_logger.log_sleep_stage_f1_scores(labels_valid['actual'], labels_valid['predicted'],
                                                                  'valid')
        for stage in f1_scores_valid:
            f1_scores['valid'][stage].append(f1_scores_valid[stage])

        new_best_model = f1_scores_valid['avg'] > best_avg_f1_score
        result_logger.log_confusion_matrix(labels_valid['actual'], labels_valid['predicted'], 'valid',
                                           wo_plot=not new_best_model)
        result_logger.log_transformation_matrix(labels_valid['actual'], labels_valid['predicted'], 'valid',
                                                wo_plot=not new_best_model)

        if new_best_model:
            best_avg_f1_score = f1_scores_valid['avg']
            best_epoch = epoch
            snapshot(config, {
                'model': config.MODEL_NAME,
                'epoch': epoch,
                'validation_avg_f1_score': f1_scores_valid['avg'],
                'state_dict': model.state_dict(),
                'clas_optimizer': optimizer.state_dict(),
            }, config.EXTRA_SAFE_MODELS)

        end = time.time()
        logger.logger.info('[epoch {:3d}] execution time: {:.2f}s\t avg f1-score: {:.4f}\n'.format(epoch, (end - start),
                                                                                                   f1_scores_valid[
                                                                                                       'avg']))

        # early stopping
        # stop training if the validation f1 score has not increased over the last 5 epochs
        # but only do so after WARMUP_EPOCHS was reached
        if epoch >= config.WARMUP_EPOCHS and epoch - best_epoch > 4:
            break

    result_logger.plot_f1_score_course(f1_scores)
    result_logger.log_metrics({'loss': losses})
    logger.fancy_log('finished training')
    logger.fancy_log('best model on epoch: {} \tf1-score: {:.4f}'.format(best_epoch, best_avg_f1_score))


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment)

    logger = Logger(config)
    logger.init_log_file(args, basename(__file__))

    logger.fancy_log('start training with model: {}'.format(config.MODEL_NAME))
    training()
