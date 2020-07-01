import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from base.config_loader import ConfigLoader
from base.evaluation.matrix_plotter import MatrixPlotter
from base.utilities import format_dictionary

logger = logging.getLogger('tuebingen')


class ResultLogger:
    def __init__(self, config: ConfigLoader):
        """ wrapper class for a logger, that logs and plot various results and metrics """
        self.config = config
        self.matrix_plotter = MatrixPlotter(self.config)

    def plot_f1_score_course(self, f1_scores):
        """ plots f1 scores against epochs

        Args:
            f1_scores (dict): dict containing the f1 scores to be plotted, it must have the two elements (dicts) 'train'
                and 'valid'. Both dicts need to at least have a nested element 'avg'. 'valid' may additionally contain
                elements for single stages.
        """
        # plot avg f1 scores for every dataset in f1_scores in a combined plot
        plt.figure()
        plt.title('avg F1 scores')
        t = np.arange(len(f1_scores['train']['avg']))
        for ds in f1_scores:
            plt.plot(t, f1_scores[ds]['avg'], label=ds)
        plt.xticks(t)
        plt.xlabel('epochs')
        plt.ylabel('F1-score')
        plt.ylim((0.5, 1))
        plt.grid()
        plt.legend()
        plt.savefig(self.config.VISUALS_DIR + '/sleep_stages_avg_f1_scores.png')

        # plot f1 scores for each stage in f1_scores['valid'] in a combined plot
        plt.figure()
        plt.title('F1 scores on all stages for dataset valid')
        for stage in f1_scores['valid']:
            plt.plot(t, f1_scores['valid'][stage], label=stage)
        plt.legend()
        plt.xticks(t)
        plt.xlabel('epochs')
        plt.ylabel('F1-score')
        plt.ylim((0.0, 1))
        plt.grid()
        plt.savefig(self.config.VISUALS_DIR + '/sleep_stages_f1_scores_valid.png')

        # log f1_scores
        np.set_printoptions(formatter={'float_kind': '{:.3f}'.format})
        for ds in f1_scores:
            logger.info('final F1-scores on ' + str(ds) + ':\n' + format_dictionary(f1_scores[ds]))
        np.set_printoptions(formatter=None)

    def log_transformation_matrix(self, actual_labels, predicted_labels, dataset, wo_plot=False):
        """ logs and plots transformation matrices if `wo_plot`is False\n
        created plots are saved as files in VISUALS_DIR

        Args:
            actual_labels (list): list containing actual labels, must be temporally sorted
            predicted_labels (list): list containing predicted labels, must be temporally sorted
            dataset (str): name of the dataset the labels are from
            wo_plot (bool): flag, whether plots are to be created and saved in a file
        """
        tm_act = self.matrix_plotter.plot_tm(actual_labels, normalize=1,
                                             title='tm for actual stages on ' + dataset, wo_plot=wo_plot)
        tm_pre = self.matrix_plotter.plot_tm(predicted_labels, normalize=1,
                                             title='tm for predicted stages on ' + dataset, wo_plot=wo_plot)

        logger.info('transformation matrix for sleep stages on dataset: {0!s}'.format(dataset))
        percent_formatter = '{:6.2%}'.format
        np.set_printoptions(formatter={'float_kind': percent_formatter})
        logger.info('actual:')
        logger.info('\n' + str(tm_act))
        logger.info('predicted:')
        logger.info('\n' + str(tm_pre))
        np.set_printoptions(formatter=None)

    def log_sleep_stage_f1_scores(self, actual_labels, predicted_labels, dataset):
        """ logs precision, recall and f1-scores for each stage and as average over all stages

        Args:
            actual_labels (list): list containing actual labels
            predicted_labels (list): list containing predicted labels
            dataset (str): name of the dataset the labels are from

        Returns:
            dict: dict containing f1-scores for every stage and the average f1-score
         """
        f1_scores = {stage: 0 for stage in self.config.STAGES + ['avg']}
        n_labels = np.arange(len(self.config.STAGES))

        logger.info('scores for sleep stages on dataset: {0!s}'.format(dataset))
        for n, stage in enumerate(self.config.STAGES):
            # the f1-score of a stage only represents, whether the stage was classified correctly which reduces the
            # problem to a binary classificationof 'current stage' vs 'other stage'
            actual = [1 if act_label == n else 0 for act_label in actual_labels]
            pred = [1 if pred_label == n else 0 for pred_label in predicted_labels]
            logger.info('precision on {0!s}: {1:.3f}'.format(stage, precision_score(actual, pred)))
            logger.info('recall on {0!s}: {1:.3f}'.format(stage, recall_score(actual, pred)))
            f1_score_n = f1_score(actual, pred)
            f1_scores[stage] = f1_score_n
            logger.info('F1-score on {0!s}: {1:.3f}'.format(stage, f1_score_n))
            logger.info('')

        logger.info('average precision on sleep stages: {0:.3f}'.format(
            precision_score(actual_labels, predicted_labels, average='macro', labels=n_labels)))
        logger.info('average recall on sleep stages: {0:.3f}'.format(
            recall_score(actual_labels, predicted_labels, average='macro', labels=n_labels)))
        f1_score_ss = f1_score(actual_labels, predicted_labels, average='macro', labels=n_labels)
        logger.info('average F1-score on sleep stages: {0:.3f}'.format(f1_score_ss))
        f1_scores['avg'] = f1_score_ss

        return f1_scores

    def log_confusion_matrix(self, actual_labels, predicted_labels, dataset, wo_plot):
        """ logs and plots confusion matrices if `wo_plot`is False\n
        created plots are saved as files in VISUALS_DIR

        Args:
            actual_labels (list): list containing actual labels
            predicted_labels (list): list containing predicted labels
            dataset (str): name of the dataset the labels are from
            wo_plot (bool): flag, whether plots are to be created and saved in a file
        """
        logger.info('confusion matrix for sleep stages on dataset: {0!s}'.format(dataset))
        # confusion matrix with absolute sample numbers
        cm = self.matrix_plotter.plot_cm(actual_labels, predicted_labels, np.array(self.config.STAGES),
                                         title='cm sleep stages abs on ' + dataset, wo_plot=True)
        logger.info('\n' + str(cm))

        percent_formatter = '{:7.2%}'.format
        np.set_printoptions(formatter={'float_kind': percent_formatter})
        # confusion matrix normalized over it's columns
        cm = self.matrix_plotter.plot_cm(actual_labels, predicted_labels, np.array(self.config.STAGES),
                                         normalize=1, title='cm sleep stages t-rel on ' + dataset, wo_plot=wo_plot)
        logger.info('\n' + str(cm))
        # confusion matrix normalized over it's rows
        cm = self.matrix_plotter.plot_cm(actual_labels, predicted_labels, np.array(self.config.STAGES),
                                         normalize=0, title='cm sleep stages p-rel on ' + dataset, wo_plot=wo_plot)
        logger.info('\n' + str(cm))
        np.set_printoptions(formatter=None)

    def log_metrics(self, metrics):
        """ logs and plots various metrics of training, currently the only supported metric is the loss function\n
        all plots are saved as files in VISUALS_DIR

        Args:
            metrics (dict): dict with metrics, each metric has it's own key, currently the only supported key is 'loss'
                and must contain a dict, with losses for 'train' and 'valid'
        """
        # loss
        plt.figure()
        plt.title('losses over all data sets')
        t = np.arange(len(metrics['loss']['train']))
        for ds in metrics['loss']:
            plt.plot(t, metrics['loss'][ds], label=ds)
        plt.xticks(t)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        plt.legend()
        plt.savefig(self.config.VISUALS_DIR + '/metrics_loss.png')
