__author__ = 'ptoth'
import config.sr_network_configuration as base_config

import unittest


class ConfigTest(unittest.TestCase):

    def test_structure(self):
        conf = base_config.get_config()
        print conf['network']

