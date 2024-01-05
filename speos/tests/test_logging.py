from speos.utils.logger import setup_logger
from speos.utils.config import Config
import unittest
import shutil
import os
from utils import TestSetup

class LoggingTest(TestSetup):

    def setUp(self) -> None:
        super().setUp()

        self.config.name = "LoggingTest"

    def test_loglevel(self):
        logger = setup_logger(self.config, __name__)
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")

        with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
            numlines = len(fp.readlines())

        self.assertEqual(numlines, 5)

    def test_flush_existing_loggers(self):
        logger = setup_logger(self.config, __name__)
        logger.info("info")
        logger.info("another info")

        with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
            numlines = len(fp.readlines())

        self.assertEqual(numlines, 0)

        # getting a new loger should make the old one flush
        logger = setup_logger(self.config, __name__ + "_new")

        with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
            numlines = len(fp.readlines())

        self.assertEqual(numlines, 2)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
