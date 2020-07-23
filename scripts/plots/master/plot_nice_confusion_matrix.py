from os.path import join, dirname

import numpy as np
import matplotlib.pyplot as plt

# values in confusion matrices in percent
cm_001 = np.array([[98.73, 0.13, 1.07, 0.01, 0.06, ],
                   [0.57, 95.51, 1.41, 2.40, 0.10, ],
                   [5.32, 0.33, 92.10, 2.12, 0.12, ],
                   [1.00, 22.39, 23.55, 52.90, 0.17, ],
                   [16.13, 2.15, 26.88, 0.00, 54.84, ]])

cmT_001 = np.array([[95.63, 1.18, 1.39, 0.15, 13.25, ],
                    [0.06, 89.70, 0.19, 6.77, 2.41, ],
                    [4.20, 2.40, 97.22, 46.10, 21.69, ],
                    [0.03, 6.62, 1.02, 46.98, 1.20, ],
                    [0.08, 0.10, 0.18, 0.00, 61.45, ]])

cm_001b = np.array([[97.45, 0.79, 1.33, 0.04, 0.39, ],
                    [0.89, 94.10, 0.47, 4.28, 0.26, ],
                    [5.93, 0.78, 88.01, 4.66, 0.62, ],
                    [1.33, 16.09, 15.59, 66.83, 0.17, ],
                    [2.15, 1.08, 5.38, 0.00, 91.40, ]])

cmT_001b = np.array([[95.14, 6.58, 1.81, 0.59, 27.95, ],
                     [0.09, 83.54, 0.07, 6.96, 1.97, ],
                     [4.71, 5.33, 97.38, 58.27, 36.22, ],
                     [0.04, 4.50, 0.71, 34.18, 0.39, ],
                     [0.01, 0.05, 0.04, 0.00, 33.46, ]])

filename = 'cm_all'

stages = ['Wake', 'REM', 'Non REM', 'Pre REM', 'Artefakt']


def plot_confusion_matrix(cm, axis, labels_x=True, labels_y=True):
    axis.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # we want to show all ticks...
    axis.set(xticks=np.arange(cm.shape[1]),
             yticks=np.arange(cm.shape[0]),
             # ... and label them with the respective stages
             xticklabels=stages, yticklabels=stages,
             # title=title,
             ylabel='wahre Klassen' if labels_y else '',
             xlabel='vorhergesagte Klassen' if labels_x else '')
    axis.set_ylim([cm.shape[0] - 0.5, -0.5])

    # rotate the tick labels and set their alignment.
    plt.setp(axis.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # loop over data dimensions and create text annotations.
    fmt = '.1%'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                      color='white' if cm[i, j] > thresh else 'black')


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(9, 8))
axes = axes.flatten()
for i, (ax, data) in enumerate(zip(axes, [cmT_001, cm_001, cmT_001b, cm_001b])):
    data /= 100.
    plot_confusion_matrix(data, ax, i >= 2, i % 2 == 0)
# save plots
fig.tight_layout()
plt.savefig(join(dirname(__file__), '../../..', 'results', 'plots', 'master', filename + '.png'))
plt.show()
