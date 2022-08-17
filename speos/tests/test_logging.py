from speos.utils.logger import setup_logger
from speos.utils.config import Config
import unittest
import shutil
import os


class LoggingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = Config()
        cls.config.logging.dir = "speos/tests/logs/"
        cls.config.logging.level = 20

        cls.config.name = "LoggingTest"

    def tearDown(self):
        shutil.rmtree(self.config.logging.dir, ignore_errors=True)

    def test_loglevel(self):
        logger = setup_logger(self.config, __name__)
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        logger.critical("critical")

        with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
            numlines = len(fp.readlines())

        self.assertEqual(numlines, 4)

    def test_flush_existing_loggers(self):
        logger = setup_logger(self.config, __name__)
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")

        #with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
        #    numlines = len(fp.readlines())

        #self.assertEqual(numlines, 0)

        # getting a new loger should make the old one flush
        logger = setup_logger(self.config, __name__ + "_new")

        with open(os.path.join(self.config.logging.dir, self.config.name), 'r') as fp:
            numlines = len(fp.readlines())

        self.assertEqual(numlines, 2)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
