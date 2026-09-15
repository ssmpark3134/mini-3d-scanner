import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import tempfile
import unittest
import numpy as np
from common import read_image, read_json
from capture import save_capture
from reconstruct import carve, project, voxel_centers
from silhouette import make_mask
from synthetic_demo import box_inputs


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.masks, self.config = box_inputs()
        self.cameras = self.config['cameras']

    def test_projection_axes(self):
        points = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
        u, v = project(points, dict(angle=0, cx=100, cy=100, scale=10))
        np.testing.assert_array_equal(u, [110, 100, 100])
        np.testing.assert_array_equal(v, [100, 100, 90])
        u, _ = project(points, dict(angle=90, cx=100, cy=100, scale=10))
        np.testing.assert_array_equal(u, [100, 90, 100])

    def test_box_and_monotonic(self):
        voxels, counts = carve(self.masks, self.cameras)
        self.assertTrue(all(a >= b for a, b in zip(counts, counts[1:])))
        points = voxel_centers(40, self.config['bounds'])
        interior = np.all(np.abs(points) < [.47, .27, .57], axis=-1)
        self.assertTrue(voxels[interior].all())
        self.assertFalse(voxels[np.abs(points[..., 2]) > .65].any())
        self.assertGreater(voxels.sum(), 0)

    def test_empty(self):
        self.masks[1][:] = 0
        self.assertEqual(carve(self.masks, self.cameras)[0].sum(), 0)

    def test_boundaries(self):
        camera = dict(angle=0, cx=0, cy=0, scale=1)
        mask = np.full((2, 2), 255, np.uint8)
        voxels, _ = carve([mask], [camera], grid=2, bounds=((-2, 2),)*3)
        expected = np.zeros((2, 2, 2), bool)
        expected[1, :, 0] = True
        np.testing.assert_array_equal(voxels, expected)

    def test_white_masks(self):
        masks = [np.full_like(mask, 255) for mask in self.masks]
        self.assertEqual(carve(masks, self.cameras)[0].sum(), 40**3)

    def test_mask_hole_and_empty(self):
        frame = np.full((50, 50), 255, np.uint8)
        self.assertFalse(make_mask(frame).any())
        frame[10:40, 10:40] = 0
        frame[20:30, 20:30] = 255
        mask = make_mask(frame)
        self.assertEqual(mask[25, 25], 0)
        self.assertEqual(mask[15, 15], 255)

    def test_save_reload_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'scan'
            first = save_capture(base, self.frames, self.masks, self.config)
            second = save_capture(base, self.frames, self.masks, self.config)
            self.assertNotEqual(first, second)
            config = read_json(first / 'config.json')
            masks = [read_image(first / f'cam{i}_mask.png') for i in range(1, 4)]
            for i in range(3):
                np.testing.assert_array_equal(read_image(first / f'cam{i+1}.png'), self.frames[i])
            np.testing.assert_array_equal(carve(masks, config['cameras'])[0], carve(self.masks, self.cameras)[0])


if __name__ == '__main__':
    unittest.main()
