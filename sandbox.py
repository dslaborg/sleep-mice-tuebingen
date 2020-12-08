import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from base.config_loader import ConfigLoader
from base.data.dataloader import TuebingenDataloader
from base.utilities import calculate_tensor_size_after_convs

if __name__ == '__main__':
    calculate_tensor_size_after_convs(640 * 3, [5] * 8, [1, 2] * 4)

    # config = ConfigLoader('exp003e', create_dirs=False)
    # dataloader = TuebingenDataloader(config, 'train', augment_data=False, balanced=False)
    # aug_dataloader = TuebingenDataloader(config, 'train', augment_data=True)
    #
    # # signal = dataloader[10][0][0]
    # # aug_signal = aug_dataloader[10][0][0]
    # # aug_signal2 = aug_dataloader[10][0][0]
    # # aug_signal3 = aug_dataloader[10][0][0]
    # # aug_signal4 = aug_dataloader[10][0][0]
    # #
    # # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    # # ax1.plot(signal)
    # # ax2.plot(aug_signal)
    # # ax3.plot(aug_signal2)
    # # ax4.plot(aug_signal3)
    # # ax5.plot(aug_signal4)
    # # plt.show()
    #
    # dl = torch.utils.data.dataloader.DataLoader(dataloader, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    #
    # labels = []
    # for d in dl:
    #     labels.extend(d[1])
    # print(np.unique(labels, return_counts=True))
