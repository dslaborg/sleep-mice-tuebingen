import numpy as np
import torch
from torch import nn

from base.config_loader import ConfigLoader


def evaluate(config: ConfigLoader, model, validationloader):
    """ evaluate model using the data given by validationloader

    Args:
        config (ConfigLoader): config of the experiment
        model (nn.Module): model to be evaluated
        validationloader (torch.utils.data.DataLoader): dataloader containing the data to be used for evaluation

    Returns:
        (dict, float): dict with actual and predicted labels, actual labels are accessible with the key 'actual' and the
            predicted labels with 'predicted'; float containing the loss of last evaluated minibatch
    """
    model = model.eval()
    predicted_labels = np.empty(0, dtype='int')
    actual_labels = np.empty(0, dtype='int')
    loss = torch.zeros(1)

    with torch.no_grad():
        for data in validationloader:
            features, labels = data
            features = features.to(config.DEVICE)
            labels = labels.long().to(config.DEVICE)

            outputs = model(features)

            criterion = nn.NLLLoss()
            loss = criterion(outputs, labels)

            _, predicted_labels_i = torch.max(outputs, dim=1)

            predicted_labels = np.r_[predicted_labels, predicted_labels_i.tolist()]
            actual_labels = np.r_[actual_labels, labels.tolist()]

    return {'actual': actual_labels, 'predicted': predicted_labels}, loss.item()
