from os.path import join, dirname

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.master.f1_score_courses import *

plt.rcParams.update({'font.size': 12})
# locale.setlocale(locale.LC_NUMERIC, "de_DE")
# plt.rcParams['axes.formatter.use_locale'] = True
fig, axes = plt.subplots(1, 2, sharey='all', figsize=(8, 4))
axes_fl = axes.flatten()
# axes_x = axes[0]
# axes_y = axes[1]

# f1_scores_arr = [f1_scores_001_train, f1_scores_003d_train, f1_scores_001_valid, f1_scores_003d_valid]
# f1_scores_arr = [f1_scores_005b_valid, f1_scores_001_valid]
f1_scores_arr = [f1_scores_005b_train, f1_scores_001_train]

# for ax, f1_scores in zip(axes_fl,
#                          [f1_scores_005b_train, f1_scores_006_train, f1_scores_006b_train, f1_scores_006c_train,
#                           f1_scores_006d_train, f1_scores_006e_train, f1_scores_005b_valid, f1_scores_006_valid,
#                           f1_scores_006b_valid, f1_scores_006c_valid,
#                           f1_scores_006d_valid, f1_scores_006e_valid]):
# for ax, f1_scores in zip(axes_fl,
#                          [f1_scores_001_train, f1_scores_003e_train]):
# for ax, f1_scores in zip(axes_fl,
#                          [f1_scores_001_valid, f1_scores_003d_valid]):
for ax, f1_scores in zip(axes_fl, f1_scores_arr):
    # for ax, f1_scores in zip(axes_fl,
    #                          [f1_scores_001_train, f1_scores_003_train, f1_scores_003b_train, f1_scores_003c_train,
    #                           f1_scores_003d_train, f1_scores_003e_train]):
    n = len(f1_scores['Wake'])
    t = np.arange(1, n + 1)
    for stage in f1_scores:
        if stage == 'avg': continue
        ax.plot(t, f1_scores[stage][:n], label=stage)
    ax.set_xlim((0, max([len(f1['Wake']) for f1 in f1_scores_arr]) + 1))
    ax.set_ylim((0.0, 1))
    ax.grid()
    # ax.vlines(best_epoch, 0, 1, linestyles="--", label="beste Epoche")

for ax in axes[-1] if len(axes.shape) == 2 else axes:
    ax.set_xlabel('epochs')
if len(axes.shape) == 2:
    for ax in axes:
        ax[0].set_ylabel('F1 score')
else:
    axes[0].set_ylabel('F1 score')
axes_fl[-1].legend()
plt.tight_layout()
fig.tight_layout()
plt.savefig(
    join(dirname(__file__), '../../..', 'results', 'plots', 'paper', 'f1_scores_balancing_all_stages_train.svg'))
plt.show()
