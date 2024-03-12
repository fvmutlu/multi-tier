import unittest
import simpy as sp
from simulator.cache import Permastore

class SampleTestCase(unittest.TestCase):
    def setUp(self):
        self.env = sp.Environment()
        self.p = Permastore(self.env, 10, [1,2,3])
    
    def test_sample_a(self):
        self.assertIsNotNone(self.p.contents)