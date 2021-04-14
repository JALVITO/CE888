import sys
sys.path.append('../src/')

import unittest

from data import data_split

class TestData(unittest.TestCase):
    def test_val_split(self):
        # Using default 0.2 value...
        train, val, _ = data_split()
        self.assertEqual(len(list(train.unbatch())), 31500)
        self.assertEqual(len(list(val.unbatch())), 7875)

        # Using custom value...
        train, val, _ = data_split(split=0.4)
        self.assertEqual(len(list(train.unbatch())), 23625)
        self.assertEqual(len(list(val.unbatch())), 15750)

if __name__ == '__main__':
    unittest.main()