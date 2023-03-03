import logging
from logging.handlers import MemoryHandler
import os


def flush_existing_loggers():
    """ This avoids i.e. handlers with 500 lines flush limit to keep their secrets until the program terminates if they do not reach that limit """
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]  # get all other loggers
    [handler.flush() for logger in loggers for handler in logger.handlers]  # flush all handlers


def setup_logger(config, name):
    """ Returns a logger called name and formats it according to config if it does not exist yet """

    flush_existing_loggers()
    logger = logging.getLogger(name)
    level = config.logging.level
    test = logger.hasHandlers()

    if logger.hasHandlers():
        logger.handlers = []

    if not os.path.exists(os.path.dirname(config.logging.dir)):
        os.makedirs(os.path.dirname(config.logging.dir))

    if config.logging.file == "auto":
        output = os.path.join(config.logging.dir, str(config.name))
    elif config.logging.file is None:
        output = "/dev/null"
    else:
        output = os.path.join(config.logging.dir, config.logging.file)

    formatter = logging.Formatter('{} %(asctime)s [%(levelname)s] %(name)s: %(message)s'.format(config.name))

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)

    fh = logging.FileHandler(output)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    logger.addHandler(MemoryHandler(capacity=500, flushLevel=30, target=sh))
    logger.addHandler(MemoryHandler(capacity=100, flushLevel=30, target=fh))

    logger.propagate = False
    logger.setLevel(level)

    return logger


class CustomLogger:
    def __init__(self, config):
        self.name = config.name

        if config.logging.file == "auto":
            self.output = os.path.join(config.logging.dir, str(config.name))
        elif config.logging.file is None:
            self.output = "/dev/null"
        else:
            self.output = os.path.join(config.logging.dir, config.logging.file)

        self.level = config.logging.level
        handlers = [MemoryHandler(500, target=logging.FileHandler(self.output)), MemoryHandler(500, target=logging.StreamHandler())]
        logging.basicConfig(level=self.level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

    def debug(self, message):
        logging.debug("{}: {}".format(self.name, message))

    def info(self, message):
        logging.info("{}: {}".format(self.name, message))

    def warning(self, message):
        logging.warning("{}: {}".format(self.name, message))

    def error(self, message):
        logging.error("{}: {}".format(self.name, message))
