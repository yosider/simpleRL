# coding: utf-8
from pathlib import Path
import shutil
import unittest
from utils.logger import Logger

class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger = Logger('test2')        

    def test_visualize(self):
        self.logger.add(y=100)
        self.logger.add(y=200)
        self.logger.add(y=300)
        self.logger.visualize()
        assert (self.logger.logdir / 'y.png').exists()

    def tearDown(self):
        shutil.rmtree(str(self.logger.logdir))

if __name__ == '__main__':
    unittest.main()
