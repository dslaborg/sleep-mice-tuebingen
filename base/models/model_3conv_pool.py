import torch.nn as nn
import torch.nn.functional as F

from base.config_loader import ConfigLoader


class Model(nn.Module):
    """ model with a feature extractor consisting of 3 conv layers with BatchNorm, ReLU activations and a final pooling
    layer, pooling each sample (of SAMPLES_LEFT, SAMPLES_RIGHT and the middle one).
    Output size is batch * 128 * (SAMPLES_LEFT * SAMPLES_RIGHT + 1).

    The classifier consists of a dropout layer with p=0.2 and a single linear layer which downsamples the input size to
    the number of stages """
    def __init__(self, config: ConfigLoader):
        super(Model, self).__init__()

        sample_length = int(config.SAMPLING_RATE * config.SAMPLE_DURATION)

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(len(config.CHANNELS)),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(len(config.CHANNELS), 128, 8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(128, 256, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.AvgPool1d(sample_length, sample_length)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * (config.SAMPLES_LEFT + config.SAMPLES_RIGHT + 1), len(config.STAGES))
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
