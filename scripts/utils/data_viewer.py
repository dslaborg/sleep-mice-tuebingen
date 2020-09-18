import argparse
import locale
import sys
from importlib import import_module
from os.path import join, dirname, realpath, isfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider, Button

sys.path.insert(0, realpath(join(dirname(__file__), '../..')))

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader

class_order = ['artifact', 'Non REM', 'pre-REM', 'REM', 'Wake']
class_order_mapping = {0: 4, 1: 3, 2: 1, 3: 2, 4: 0}


class DataViewer:
    def __init__(self, config: ConfigLoader, dataset: str):
        # two configs, because the model does not necessarily need all CHANNELS, but we want to see all of them
        # also the model may need additional samples to the left and right of the current sample,
        # see SAMPLES_LEFT and SAMPLES_RIGHT
        # therefore the standard_config is used for the viewer with no samples to the left and right
        # and the config from the selected experiment is used for the model
        self.view_config = ConfigLoader(create_dirs=False)
        self.view_config.SAMPLES_LEFT, self.view_config.SAMPLES_RIGHT = 0, 0
        self.model_config = config
        self.window_size = 1  # number of samples in window
        self.cur_sample = 0  # current sample

        # always load best model of selected experiment
        model_file = join(self.model_config.EXPERIMENT_DIR, self.model_config.MODEL_NAME + '-best.pth')
        if isfile(model_file):
            self.model = import_module('.' + self.model_config.MODEL_NAME, 'base.models').Model(
                self.model_config).eval()
            self.model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'])

        # number of datapoints in total window
        self.window_points = self.window_size * int(self.view_config.SAMPLE_DURATION) * self.view_config.SAMPLING_RATE
        # two dataloaders for the same reasons two configs are needed
        self.datamap_view = TuebingenDataloader(self.view_config, dataset, False, False)
        self.datamap_model = TuebingenDataloader(self.model_config, dataset, False, False)
        self.lines = {}  # lines need to be saved, so they can be updated later on
        self.axes = None
        self.fig = None

    def run(self):
        plt.rcParams.update({'font.size': 14})
        plt.rcParams['axes.formatter.use_locale'] = True

        # create an axis for every CHANNEL + an axis for the labels + an axis for the predicted probabilities
        self.fig, self.axes = plt.subplots(len(self.view_config.CHANNELS) + 2, sharex='all')
        plt.subplots_adjust(bottom=0.18, top=0.95, hspace=0.2)

        # sliders to set window size and current sample are placed under the axes for data, etc
        axcolor = 'lightgoldenrodyellow'
        ax_sample = plt.axes([0.25, 0.04, 0.65, 0.02], facecolor=axcolor)
        ax_window = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)
        slider_sample = Slider(ax_sample, 'sample', 1, len(self.datamap_view) - 12, valinit=5015, valstep=1,
                               valfmt='%i')
        slider_window = Slider(ax_window, 'window-size', 1, 10, valinit=1, valstep=1, valfmt='%i samples')

        def update_slider_sample(val):
            self.cur_sample = int(val)
            self.update_axes()

        def update_slider_window(val):
            self.window_size = int(val)
            for ax in self.axes:
                ax.set_xlim((0, self.window_size))
            self.update_axes()

        slider_sample.on_changed(update_slider_sample)
        slider_window.on_changed(update_slider_window)

        # buttons next and last to navigate through samples
        nextax = plt.axes([0.93, 0.6, 0.03, 0.04])
        next_button = Button(nextax, 'next', color=axcolor, hovercolor='0.975')
        lastax = plt.axes([0.04, 0.6, 0.03, 0.04])
        last_button = Button(lastax, 'last', color=axcolor, hovercolor='0.975')

        def next_action(event):
            self.cur_sample += 1
            slider_sample.set_val(self.cur_sample)

        def last_action(event):
            self.cur_sample -= 1
            slider_sample.set_val(self.cur_sample)

        next_button.on_clicked(next_action)
        last_button.on_clicked(last_action)

        # load initial view
        t = np.linspace(0, self.window_size, self.window_points)  # x axis for initial data view
        init_data = self.datamap_view[self.cur_sample]  # initial data shown
        # all data in dataloader, just needed to set limits for y-axis
        data_ges = np.r_[[self.datamap_view[i][0] for i in range(len(self.datamap_view))]]
        # configure first axes for data from CHANNELs
        for i, c in enumerate(self.view_config.CHANNELS):
            colors = {'EEG_FR': 'darkgreen', 'EEG1': 'darkgreen',
                      'EEG_PR': 'green',
                      'EEG_PL': 'limegreen', 'EEG2': 'limegreen',
                      'EMG': 'darkred', 'EMG1': 'darkred'}
            self.lines[c], = self.axes[i].plot(t, init_data[0][i], c=colors[c])
            minmax = np.max([-np.min(data_ges[5:, i, :]), np.max(data_ges[5:, i, :])])
            if c == 'EEG_FR': minmax = 0.4
            if c == 'EEG_PR': minmax = 0.5
            if c == 'EEG_PL': minmax = 0.005
            if c == 'EMG': minmax = 0.1
            self.axes[i].set_ylim((-minmax, minmax))
            self.axes[i].set(ylabel=c + '\n[\u03BCV]')
            # self.axes[i].set_yticks([self.axes[i].get_yticks()[0], 0, self.axes[i].get_yticks()[-1]])
            self.axes[i].grid()

        # next-to-last axis shows a hypnogram for the true labels of the samples
        self.lines['scores'], = self.axes[-2].plot(t, np.array(
            [class_order_mapping[i] for i in (init_data[1] if self.window_size != 1 else [init_data[1]])]).repeat(
            self.window_points), c='black', lw=3)
        self.axes[-2].set_ylim((-0.1, len(self.view_config.STAGES) - 0.5))
        self.axes[-2].set(yticks=np.arange(len(self.view_config.STAGES)), yticklabels=class_order)
        self.axes[-2].set(ylabel='classes\n')
        self.axes[-2].grid()

        # initial loading of last acis with predictions
        self.update_prediction_axis()

        # show data viewer maximized
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        self.fig.align_ylabels(self.axes)
        plt.show()

    def update_axes(self):
        """ update data in axes """
        t = np.linspace(0, self.window_size, self.window_size * self.window_points)  # x-axis
        # load data for each sample in window
        data = np.r_[[self.datamap_view[self.cur_sample + i] for i in range(self.window_size)]]
        signals = np.concatenate([data[i][0] for i in range(self.window_size)], axis=1)
        scores = np.array([class_order_mapping[data[i][1]] for i in range(self.window_size)]).repeat(self.window_points)
        # update CHANNEL axes and axis with labels
        for i, c in enumerate(self.view_config.CHANNELS):
            self.lines[c].set_data(t, signals[i])
        self.lines['scores'].set_data(t, scores)

        # lastly update predictions
        self.update_prediction_axis()

    def update_prediction_axis(self):
        """ update axis with predictions """
        if self.model is None:
            return None
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        self.axes[-1].clear()
        # load data from model dataloader
        data_model = torch.tensor(np.r_[[self.datamap_model[self.cur_sample + i][0] for i in range(self.window_size)]])
        # predict samples and calculate softmax-probabilities
        data_pred = np.exp(self.model(data_model).detach()[:, :len(self.view_config.STAGES)].reshape(-1).numpy())
        # create ankers on x-axis for histogram bins, one additional empty bin to the left and right for spacing
        x_coords = np.linspace(0, 1, len(self.view_config.STAGES) + 2)[1:-1]
        # create bins
        self.axes[-1].bar(np.r_[[x_coords + i for i in range(self.window_size)]].flatten(), data_pred,
                          width=1. / (len(self.view_config.STAGES) * 2),
                          color=colors[:len(self.view_config.STAGES)] * self.window_size)
        self.axes[-1].set(xlim=(0, self.window_size), ylim=(0, 1.1))
        self.axes[-1].set_xticks(np.arange(0, self.window_size + 1))
        self.axes[-1].set(xlabel='epoch', ylabel='predicted\nclass\nprobabilities')
        # create legend for bins and move outside of the figure
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(self.view_config.STAGES))]
        # self.axes[-1].legend(handles, self.view_config.STAGES[:-1] + ['Artefakt'], loc=(0.92, 0.0))
        self.axes[-1].legend(handles, reversed(class_order), loc=(1.01, 0.0))
        # show borders between samples using vertical lines
        for i in range(self.window_size): self.axes[-1].vlines(i, 0, 1.1, color='darkgrey')


def parse():
    parser = argparse.ArgumentParser(description='data viewer')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to load config from')
    parser.add_argument('--dataset', '-d', default='valid',
                        help='dataset to load data from')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment, create_dirs=False)
    # locale.setlocale(locale.LC_NUMERIC, "de_DE")
    DataViewer(config, args.dataset).run()
