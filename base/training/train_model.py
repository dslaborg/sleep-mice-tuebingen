import logging
from os.path import join

import numpy as np
import torch
import torch.nn as nn

from base.config_loader import ConfigLoader
from base.training.scheduled_optim import ScheduledOptim

logger = logging.getLogger('tuebingen')


def train(config, epoch, model, optimizer, trainloader):
    """train `model` using the data from `trainloader`

    Args:
        config (ConfigLoader): configuration of the experiment
        epoch (int): epoch that is trained (just for logging purposes)
        model (nn.Module): model to be trained
        optimizer (ScheduledOptim): scheduled optimizer containing the optimizer used for training
        trainloader (torch.utils.data.DataLoader): dataloader with the data to train on

    Returns:
        (dict, float): dict with 'actual' and 'predicted' labels and loss of last minibatch
    """
    model.train()  # just as a precaution, so BN and dropout are active
    predicted_labels = np.empty(0, dtype='int')  # save predicted and actual labels to return
    actual_labels = np.empty(0, dtype='int')
    # frequency of logs (every LOG_INTERVAL% of data in trainloader)
    log_fr = max(int(config.LOG_INTERVAL / 100. * len(trainloader)), 1)
    loss = torch.zeros(1)

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        features, labels = data
        features = features.to(config.DEVICE)
        labels = labels.long().to(config.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(features)

        criterion = nn.NLLLoss()
        loss = criterion(outputs, labels)

        # L1 regularization
        reg_loss = 0
        for param in model.parameters():
            reg_loss += torch.sum(torch.abs(param))
        loss += config.L1_WEIGHT_DECAY * reg_loss

        loss.backward()
        # clip the gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 0.1, 'inf')

        optimizer.step()
        lr = optimizer.update_learning_rate()  # update lr after each step (minibatch)

        # determine predicted labels and save them together with actual labels
        _, predicted_labels_i = torch.max(outputs, dim=1)
        predicted_labels = np.r_[predicted_labels, predicted_labels_i.tolist()]
        actual_labels = np.r_[actual_labels, labels.tolist()]

        # log various information every log_fr minibatches
        if i % log_fr == log_fr - 1:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.2e}\tLoss: {:.6f}'.format(
                epoch, i * len(features), len(trainloader.dataset),
                       100. * i / len(trainloader), lr, loss.item()))

    return {'actual': actual_labels, 'predicted': predicted_labels}, loss.item()


def snapshot(config: ConfigLoader, model_state, extra_safe=False):
    """save a snapshot of the model_state in EXPERIMENT_DIR

    if extra_safe is set to True, an additional snapshot is saved in MODELS_DIR (useful because the name for the
    standard snapshot is somewhat generic and the file would be overwritten with a new run of the exp"""
    snapshot_file = join(config.EXPERIMENT_DIR, config.MODEL_NAME + '-best.pth')
    torch.save(model_state, snapshot_file)
    logger.info('snapshot saved to {}'.format(snapshot_file))

    if extra_safe:
        add_snapshot_file = join(config.MODELS_DIR, config.MODEL_NAME + '-' + config.RUN_NAME + '.pth')
        torch.save(model_state, add_snapshot_file)
        logger.info('additional snapshot saved to {}'.format(add_snapshot_file))
