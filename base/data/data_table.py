import tables

from base.config_loader import ConfigLoader

COLUMN_MOUSE_ID = 'mouse_id'
COLUMN_LABEL = 'label'


def create_table_description(config: ConfigLoader):
    """ creates the description for the pytables table used for dataloading """
    n_sample_values = int(config.SAMPLING_RATE * config.SAMPLE_DURATION)

    table_description = {
        COLUMN_MOUSE_ID: tables.Int16Col(),
        COLUMN_LABEL: tables.StringCol(10)
    }
    for c in config.CHANNELS:
        table_description[c] = tables.Float32Col(shape=n_sample_values)

    return table_description
