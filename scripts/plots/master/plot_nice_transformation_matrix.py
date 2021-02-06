from os.path import join, dirname

import matplotlib.pyplot as plt
import numpy as np

# values in transformation matrices in percent
tm_test = np.array([[95.23, 0.00, 4.74, 0.00, 0.04, ],
                    [16.50, 79.58, 3.45, 0.21, 0.26, ],
                    [3.57, 0.07, 92.69, 3.33, 0.35, ],
                    [2.99, 62.85, 16.09, 17.91, 0.17, ],
                    [2.15, 2.15, 63.44, 1.08, 31.18, ]])
tm_001_test = np.array([[94.29, 0.16, 5.37, 0.04, 0.14, ],
                        [17.70, 77.45, 4.22, 0.48, 0.15, ],
                        [4.49, 0.29, 90.74, 4.07, 0.42, ],
                        [3.27, 45.86, 19.25, 31.62, 0.00, ],
                        [24.43, 2.29, 38.93, 0.76, 33.59, ]])

tm_orig = np.array([[95.72, 4.26, 0.01, 0.0], [4.04, 92.35, 3.54, 0.07], [2.46, 17.43, 27.84, 52.27], [15.48, 4.14, 0.27, 80.11]])
tm_proc = np.array([[95.72, 4.27, 0.01, 0.0], [4.07, 93.35, 2.58, 0.0], [0.0, 0.0, 0.0, 100.0], [15.46, 4.22, 0.21, 80.11]])

# stages = ['Wake', 'REM', 'NREM', 'pre-REM']  # , 'Artefakt']
stages = ['Wake', 'NREM', 'pre-REM', 'REM']  # , 'Artefakt']
letters = ['a', 'b', 'c', 'd', 'e', 'f']


def plot_transformation_matrix(tm, axis, labels_x=True, labels_y=True):
    axis.imshow(tm, interpolation='nearest', cmap=plt.cm.Blues)
    # we want to show all ticks...
    axis.set(xticks=np.arange(tm.shape[1]),
             yticks=np.arange(tm.shape[0]),
             # ... and label them with the respective stages
             xticklabels=stages, yticklabels=stages,
             # title=title,
             ylabel='transition from sleep stage' if labels_y else '',
             xlabel='to sleep stage' if labels_x else '')
    axis.set_ylim([tm.shape[0] - 0.5, -0.5])

    # rotate the tick labels and set their alignment.
    plt.setp(axis.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # loop over data dimensions and create text annotations.
    fmt = '.1%'
    thresh = tm.max() / 2.
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            axis.text(j, i, format(tm[i, j], fmt), ha='center', va='center',
                      color='white' if tm[i, j] > thresh else 'black')


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(9, 5))
axes = axes.flatten()
for i, (ax, data, letter) in enumerate(zip(axes, [tm_orig, tm_proc], letters)):
    tm_cut = data[:4, :4] / 100.
    tm_cut /= np.sum(tm_cut, axis=1)
    plot_transformation_matrix(tm_cut, ax, True, i % 2 == 0)
    ax.text(0.5, 1.05, f'{letter})',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
# save plots
fig.tight_layout()
plt.savefig(join(dirname(__file__), '../../..', 'results', 'plots', 'paper', 'postprocessing.svg'))
plt.show()
