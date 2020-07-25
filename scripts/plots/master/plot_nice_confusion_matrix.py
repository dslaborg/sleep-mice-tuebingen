from os.path import join, dirname

import numpy as np
import matplotlib.pyplot as plt

# values in confusion matrices in percent
cm_001 = np.array([[98.09, 0.32, 1.43, 0.02, 0.14, ],
                   [0.78, 94.20, 1.04, 3.92, 0.05, ],
                   [4.49, 0.52, 91.83, 2.91, 0.26, ],
                   [1.16, 20.23, 20.40, 58.04, 0.17, ],
                   [9.68, 2.15, 17.20, 0.00, 70.97, ]])

cmT_001 = np.array([[96.24, 2.81, 1.86, 0.47, 19.08, ],
                    [0.08, 87.49, 0.14, 8.75, 0.76, ],
                    [3.59, 3.69, 97.00, 49.94, 29.01, ],
                    [0.04, 5.92, 0.88, 40.84, 0.76, ],
                    [0.05, 0.10, 0.11, 0.00, 50.38, ]])

cm_001b = np.array([[97.51, 0.66, 1.50, 0.05, 0.28, ],
                    [0.68, 93.00, 1.78, 4.33, 0.21, ],
                    [5.23, 0.52, 90.75, 3.22, 0.28, ],
                    [0.66, 17.08, 23.55, 58.37, 0.33, ],
                    [6.45, 1.08, 16.13, 0.00, 76.34, ]])

cmT_001b = np.array([[95.69, 5.72, 1.96, 0.98, 30.18, ],
                     [0.07, 85.62, 0.25, 9.04, 2.37, ],
                     [4.19, 3.65, 96.66, 51.63, 24.26, ],
                     [0.02, 4.95, 1.03, 38.34, 1.18, ],
                     [0.03, 0.05, 0.11, 0.00, 42.01, ]])

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
