import torch.nn as nn
import torch.nn.functional as F

from base.config_loader import ConfigLoader
from base.utilities import calculate_tensor_size_after_convs


class Model(nn.Module):
    """ model as it is described in https://arxiv.org/abs/1809.08443

    feature extractor with 8 conv layers, BatchNorm, ReLU activations and dropout layers; the dropout probabilities can
    be set in the config file of the experiment (FEATURE_EXTR_DROPOUT)

    classifier consists of 2 dropout layers followed by linear layers with a ReLU activation after the first layer; the
    dropout probabilities can be set in the config file of the experiment as well (CLASSIFIER_DROPOUT)
    """

    def __init__(self, config: ConfigLoader):
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(len(config.CHANNELS)),
            nn.Conv1d(len(config.CHANNELS), config.FILTERS, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[0]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[1]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[2]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[3]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[4]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[5]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[6]),

            nn.Conv1d(config.FILTERS, config.FILTERS, 5, 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.FILTERS),
            nn.Dropout(p=config.FEATURE_EXTR_DROPOUT[7])
        )

        sample_length = config.SAMPLING_RATE * config.SAMPLE_DURATION
        self.fc1_size = calculate_tensor_size_after_convs(
            sample_length * (config.SAMPLES_LEFT + config.SAMPLES_RIGHT + 1),
            [5] * 8, [1, 2] * 4)
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.CLASSIFIER_DROPOUT[0]),
            nn.Linear(config.FILTERS * self.fc1_size, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.CLASSIFIER_DROPOUT[1]),
            nn.Linear(80, len(config.STAGES))
        )

        # initialize weights using He-initialization
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.apply(_weights_init)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        y = F.log_softmax(x, dim=1)
        return y
