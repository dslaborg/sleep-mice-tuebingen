import argparse
import sys
from os.path import dirname, join, realpath

import tables

sys.path.insert(0, realpath(join(dirname(__file__), '../..')))

from base.config_loader import ConfigLoader
from base.data.data_table import COLUMN_MOUSE_ID


def parse():
    parser = argparse.ArgumentParser(description='script to count samples per dataset')
    parser.add_argument('--experiment', '-e', required=True,
                        help='name of experiment to load config from')

    return parser.parse_args()


def get_mouse_from_id(mouse_id):
    return 'M' + str(mouse_id)[:2]


def main():
    datasets = ['train', 'valid', 'test']

    with tables.open_file(config.DATA_FILE) as file:

        for ds in datasets:
            table = file.root[ds]
            n_total = len(table[:])

            print('dataset:', ds)
            print('mice:', set(map(get_mouse_from_id, table[:][COLUMN_MOUSE_ID])))
            print('{:12s}{}\t{}'.format('stage', 'relative', 'total'))
            for s, k in config.STAGE_MAP.items():
                n = len([row[COLUMN_MOUSE_ID] for row in table.where('(label=="{}")'.format(k))])
                print('{:12s}{:6.2%}\t{:6d}'.format(s, n / n_total, n))
            print()


if __name__ == '__main__':
    args = parse()
    config = ConfigLoader(args.experiment)

    main()
