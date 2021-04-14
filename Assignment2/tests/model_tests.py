import sys
sys.path.append('../src/')

import unittest
import numpy as np
from tensorflow.keras import Sequential

from model import Brightness, Contrast, Saturation

class TestAugmentations(unittest.TestCase):
    def test_contrast(self):
        # Increase contrast in a two-colored image using contrast formula
        augment = Sequential([
            Contrast(2, input_shape=(2,2,1)),
        ])
        orig_img = np.array([[[1], [1]],[[2], [2]]])
        aug_img = np.squeeze(augment(np.array([orig_img])).numpy())
        expected_img = np.squeeze(np.array([[[0.5], [0.5]],[[2.5], [2.5]]]))

        self.assertTrue(np.array_equal(aug_img, expected_img))

    def test_brightness(self):
        # Brighten a black pixel by 0.1
        augment = Sequential([
            Brightness(0.1, input_shape=(1,1,1)),
        ])
        orig_img = np.zeros((1,1,1))
        aug_img = np.squeeze(augment(np.array([orig_img])).numpy())

        self.assertAlmostEqual(aug_img.item(0), 0.1)

    def test_saturation(self):
        # Increase saturation for a mostly-blue pixel using saturation formula
        augment = Sequential([
            Saturation(1.5, input_shape=(1,1,3)),
        ])
        orig_img = np.array([[[50, 100, 150]]])
        aug_img = np.squeeze(augment(np.array([orig_img])).numpy())
        expected_img = np.squeeze(np.array([[[0, 75, 150]]]))

        self.assertTrue(np.array_equal(aug_img, expected_img))

if __name__ == '__main__':
    unittest.main()