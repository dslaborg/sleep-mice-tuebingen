import logging
import os

from base.config_loader import ConfigLoader


class Logger:
    def __init__(self, config):
        """
        creates a logger that writes logs to the console and to a file

        the file can be found in `LOG_DIR`

        the logger is registered under the name `tuebingen`

        Args:
            config (ConfigLoader): ConfigLoader containing the parameters from your config file
        """
        self.config = config
        # initialize logger
        self.logger = logging.getLogger('tuebingen')
        self.logger.setLevel(logging.INFO)

        # create the logging file handler
        log_file = os.path.join(config.EXPERIMENT_DIR, config.RUN_NAME + '.log')
        fh = logging.FileHandler(log_file)

        # create the logging console handler
        ch = logging.StreamHandler()

        # format
        formatter = logging.Formatter('[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add handlers to logger object
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def init_log_file(self, args, scriptname: str):
        """logs some initial info about the run, like the arguments given to the script, the scriptname and the complete
        config specified by the chosen experiment"""
        self.logger.info(scriptname + ', args: ' + str(args) + '\n' + '=' * 80)
        self.logger.info('config:')
        self.logger.info('{:20s}| {}'.format('parameter name', 'parameter value'))
        list(filter(lambda x: self.logger.info('{:20s}: {}'.format(x, self.config.__dict__[x])), self.config.__dict__))

    def fancy_log(self, msg: str):
        """just a method to highlight special log messages"""
        self.logger.info('')
        self.logger.info('=' * 80)
        self.logger.info(msg)
        self.logger.info('=' * 80)
        self.logger.info('')
