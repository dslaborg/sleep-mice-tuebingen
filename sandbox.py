import tables

from base.config_loader import ConfigLoader
from base.data.data_table import COLUMN_MOUSE_ID, COLUMN_LABEL
from base.data.dataloader import TuebingenDataloader

if __name__ == '__main__':
    config = ConfigLoader()
    config.SAMPLES_LEFT, config.SAMPLES_RIGHT = 0, 0
    tables = tables.open_file(config.DATA_FILE)
    data = tables.root['train']
    print(data[:][COLUMN_MOUSE_ID])
    print(data[:][COLUMN_LABEL])
    print(data.get_where_list('(label=={})'.format(b'Artifact')))
    print(data.get_where_list('(label=="{}")'.format('Artifact')))
    print(data.get_where_list('(label=={})'.format(bytes(config.STAGES[0], 'utf-8'))))
    print(data.get_where_list('(label=="{}")'.format(config.STAGES[0])))

    dl = TuebingenDataloader(config, 'train', False, False)
    print(dl[0])
    print(dl[1])
    print(dl[2])
