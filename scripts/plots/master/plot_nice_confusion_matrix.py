from os.path import join, dirname

import numpy as np
import matplotlib.pyplot as plt

# values in confusion matrices in percent
cm2 = np.array([[98.73, 0.13, 1.07, 0.01, 0.06, ],
                [0.57, 95.51, 1.41, 2.40, 0.10, ],
                [5.32, 0.33, 92.10, 2.12, 0.12, ],
                [1.00, 22.39, 23.55, 52.90, 0.17, ],
                [16.13, 2.15, 26.88, 0.00, 54.84, ]])

cm = np.array([[95.63, 1.18, 1.39, 0.15, 13.25, ],
               [0.06, 89.70, 0.19, 6.77, 2.41, ],
               [4.20, 2.40, 97.22, 46.10, 21.69, ],
               [0.03, 6.62, 1.02, 46.98, 1.20, ],
               [0.08, 0.10, 0.18, 0.00, 61.45, ]])

cmb = np.array([[97.45, 0.79, 1.33, 0.04, 0.39, ],
                [0.89, 94.10, 0.47, 4.28, 0.26, ],
                [5.93, 0.78, 88.01, 4.66, 0.62, ],
                [1.33, 16.09, 15.59, 66.83, 0.17, ],
                [2.15, 1.08, 5.38, 0.00, 91.40, ]])

cmbT = np.array([[95.14, 6.58, 1.81, 0.59, 27.95, ],
                 [0.09, 83.54, 0.07, 6.96, 1.97, ],
                 [4.71, 5.33, 97.38, 58.27, 36.22, ],
                 [0.04, 4.50, 0.71, 34.18, 0.39, ],
                 [0.01, 0.05, 0.04, 0.00, 33.46, ]])

filename = 'exp001_cmT_test'

cm /= 100.

stages = ['Wake', 'REM', 'Non REM', 'Pre REM', 'Artefakt']

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# we want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective stages
       xticklabels=stages, yticklabels=stages,
       # title=title,
       ylabel='wahre Klassen',
       xlabel='vorhergesagte Klassen')
plt.ylim(cm.shape[0] - 0.5, -0.5)

# rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# loop over data dimensions and create text annotations.
fmt = '.1%'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black')
fig.tight_layout()
fig.subplots_adjust(left=0, bottom=0.2, right=1, top=0.99)
# save plots
plt.savefig(join(dirname(__file__), '../../..', 'results', 'plots', 'master', filename + '.png'))
plt.show()
