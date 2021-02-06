import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from base.config_loader import ConfigLoader


class MatrixPlotter:
    def __init__(self, config: ConfigLoader):
        """ class to plot confusion and transformation matrix """
        self.config = config

    def plot_cm(self, y_true, y_pred, stages, normalize=None, title=None, cmap=plt.cm.Blues, wo_plot=False):
        """
        This function prints and plots the confusion matrix specified by `y_true` and `y_pred`.
        Normalization can be applied by setting `normalize`. The plot is also saved as a file in VISUALS_DIR.

        Args:
            y_true (list or np.ndarray): actual labels
            y_pred (list or np.ndarray): predicted labels
            stages (np.ndarray): all possible stages y_true or y_pred can assume
            normalize (int): axis to normalize over
            title (str): title of the plot, is also used as the name of the file this plot is save in
            cmap (matplotlib.colors.Colormap or str): color theme of the plot
            wo_plot (bool): if set to True, no plot is generated
        """

        if title is None:
            if normalize is not None:
                title = 'normalized confusion matrix'
            else:
                title = 'confusion matrix, without normalization'

        # compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # only use the labels that appear in the data
        stages = stages[unique_labels(y_true, y_pred)]
        # optionally normalize either using x- or y-axis
        if normalize == 1:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 0:
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]

        if wo_plot:
            return cm

        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # we want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective stages
               xticklabels=stages, yticklabels=stages,
               title=title,
               ylabel='true label',
               xlabel='predicted label')
        plt.ylim(cm.shape[0] - 0.5, -0.5)

        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # loop over data dimensions and create text annotations.
        fmt = '.1%' if normalize is not None else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        fig.tight_layout()
        # save plot in VISUALS_DIR
        plt.savefig(self.config.VISUALS_DIR + '/' + title.replace(' ', '_') + '.png')
        plt.close()
        return cm

    def plot_tm(self, data, normalize=1, title=None, cmap=plt.cm.Blues, wo_plot=False):
        """
        This function prints and plots the transformation matrix specified by `y_true` and `y_pred`.
        Normalization can be applied by setting `normalize`. The plot is also saved as a file in VISUALS_DIR.

        Args:
            data (list or np.ndarray): list of labels, must be temporally ordered
            normalize (int): axis to normalize over
            title (str): title of the plot, is also used as the name of the file this plot is save in
            cmap (matplotlib.colors.Colormap or str): color theme of the plot
            wo_plot (bool): if set to True, no plot is generated
        """

        if title is None:
            if normalize is not None:
                title = 'normalized transformation matrix'
            else:
                title = 'transformation matrix without normalization'

        # compute transformation matrix from the ordered data
        # tm = np.zeros((len(self.config.STAGES), len(self.config.STAGES)), dtype='int')
        d = max(data) + 1
        tm = np.zeros((d, d), dtype='int')
        for i, stage in enumerate(data):
            if i == 0:
                continue
            tm[data[i - 1], stage] += 1

        # optionally normalize matrix either using x- or y-axis
        if normalize == 1:
            tm = tm.astype('float') / tm.sum(axis=1)[:, np.newaxis]
        elif normalize == 0:
            tm = tm.astype('float') / tm.sum(axis=0)[np.newaxis, :]

        if wo_plot:
            return tm

        fig, ax = plt.subplots()
        ax.imshow(tm, interpolation='nearest', cmap=cmap)
        # we want to show all ticks...
        ax.set(xticks=np.arange(tm.shape[1]),
               yticks=np.arange(tm.shape[0]),
               # ... and label them with the respective stages
               xticklabels=self.config.STAGES, yticklabels=self.config.STAGES,
               title=title,
               ylabel='from stage',
               xlabel='to stage')
        plt.ylim(tm.shape[0] - 0.5, -0.5)

        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 rotation_mode='anchor')

        # loop over data dimensions and create text annotations.
        fmt = '.1%' if normalize is not None else 'd'
        thresh = tm.max() / 2.
        for i in range(tm.shape[0]):
            for j in range(tm.shape[1]):
                ax.text(j, i, format(tm[i, j], fmt), ha='center', va='center',
                        color='white' if tm[i, j] > thresh else 'black')
        fig.tight_layout()
        # save plot in VISUALS_DIR
        plt.savefig(self.config.VISUALS_DIR + '/' + title.replace(' ', '_') + '.png')
        # plt.show()
        plt.close()
        return tm
