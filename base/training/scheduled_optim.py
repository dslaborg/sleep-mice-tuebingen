import numpy as np
from torch import optim


class ScheduledOptim(object):
    def __init__(self, optimizer, peak_lr, warmup_epochs, total_epochs, parameters, mode='exp'):
        """wrapper class for a pytorch optimizer with learning rate scheduling

        possible modes are:\n
        * 'step': lr = peak_lr * (1. / (1. + hp * current_steps))\n
        * 'half': lr = peak_lr * hp[0] ** floor((1 + epoch) / hp[1])\n
        * 'exp': lr = peak_lr * exp(-hp * epochs)\n
        * 'plat': lr = peak_lr with warmup and warmoff\n
        * None: lr = peak_lr w/o warmup

        Args:
            optimizer (optim.optimizer.Optimizer): pytorch optimizer whose lr should be scheduled
            peak_lr (float): max learning rate after warmup
            warmup_epochs (int): number of epochs for warmup phase; during warmup the lr is linearly increased from
                `peak_lr` / 10 to `peak_lr`
            total_epochs (int): number of total epochs the scheduler should be active; after `total_epochs` is reached
                the lr remains constant
            parameters (list or None): parameters for the selected mode, see `mode` for more details; if no parameters
                are needed None is also accepted
            mode (str or None): mode the scheduler uses to decrease lr after warmup
        """
        self.optimizer = optimizer
        self.n_current_steps = 0
        self.peak_lr = peak_lr
        self.current_lr = peak_lr / 10.
        self.start_lr = self.current_lr
        self.mode = mode
        self.current_epochs = 0
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.hp = parameters
        self.update_learning_rate()

    def state_dict(self):
        """get state_dict of inner optimizer"""
        self.optimizer.state_dict()

    def step(self):
        """step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """zero out the gradients of the inner optimizer"""
        self.optimizer.zero_grad()

    def inc_epoch(self):
        """increase epoch by 1"""
        self.current_epochs += 1

    def update_learning_rate(self):
        """Learning rate scheduling per step

        Returns:
            float: updated learning rate
        """

        # no mode results in a constant lr
        if self.mode is None:
            self.current_lr = self.peak_lr

        # warmup phase
        elif self.current_epochs <= self.warmup_epochs:
            self.current_lr = self.start_lr + (self.current_epochs - 1) * (
                    self.peak_lr - self.start_lr) / (self.warmup_epochs - 1)
        # once total_epochs is reached the lr is constant
        elif self.current_epochs > self.total_epochs:
            self.current_lr = self.current_lr
        else:
            if self.mode == 'step':
                self.n_current_steps += 1  # only update steps after warmup
                self.current_lr = self.peak_lr * (1. / (1. + self.hp[0] * self.n_current_steps))
            elif self.mode == 'exp':
                self.current_lr = self.peak_lr * np.exp(-self.hp[0] * (self.current_epochs - self.warmup_epochs))
            elif self.mode == 'half':
                self.current_lr = self.peak_lr * np.power(self.hp[0], np.floor(
                    (self.current_epochs - self.warmup_epochs) / self.hp[1]))
            elif self.mode == 'plat' and self.current_epochs > (self.total_epochs - self.warmup_epochs):
                self.current_lr = self.start_lr + (self.total_epochs - self.current_epochs) * (
                        self.peak_lr - self.start_lr) / (self.warmup_epochs - 1)

        # set lr in inner optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        return self.current_lr


# just a little program to visualize different lr courses and the effects of the modes and their parameters
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.optim import Adam
    from base.config_loader import ConfigLoader
    from base.models.model_8conv_2fc import Model

    config = ConfigLoader(create_dirs=False)

    epochs = 20
    steps_per_epoch = 1500
    # model and optimizer do not matter
    optims = {
        'exp [0.5]': ScheduledOptim(Adam(Model(config).parameters()), 1e-6 * config.BATCH_SIZE, epochs // 4, epochs,
                                    [0.5], 'exp'),
        'plat': ScheduledOptim(Adam(Model(config).parameters()), 1e-6 * config.BATCH_SIZE, epochs // 4, epochs, None,
                               'plat'),
        'None': ScheduledOptim(Adam(Model(config).parameters()), 1e-6 * config.BATCH_SIZE, epochs // 4, epochs, None,
                               None),
        'step [0.1]': ScheduledOptim(Adam(Model(config).parameters()), 1e-6 * config.BATCH_SIZE, epochs // 4, epochs,
                                     [0.1], 'step'),
        'half [0.5, 1]': ScheduledOptim(Adam(Model(config).parameters()), 1e-6 * config.BATCH_SIZE, epochs // 4, epochs,
                                        [0.5, 1], 'half')
    }

    # if set to True, steps_per_epoch updates per epoch are visualized
    # if set to False, only one update per epoch is shown (mostly only relevant for mode=step)
    all_updates = False

    y = [[] for o in optims]
    for ep in range(epochs):
        for o in optims:
            optims[o].inc_epoch()
        t_y = [[] for o in optims]
        for cs in range(steps_per_epoch):
            for i, o in enumerate(optims):
                t_y[i].append(optims[o].update_learning_rate())
        for i, o in enumerate(optims):
            if all_updates:
                y[i].extend(t_y[i])
            else:
                y[i].append(np.mean(t_y[i]))

    for i, o in enumerate(optims):
        plt.semilogy(np.arange(1, len(y[i]) + 1), y[i], label=o)

    plt.title('lr course over {} epochs with {} updates per epoch'.format(epochs, steps_per_epoch))
    plt.xlabel('updates' if all_updates else 'epochs')
    plt.ylabel('learning rate')
    plt.grid()
    plt.legend()
    plt.show()
