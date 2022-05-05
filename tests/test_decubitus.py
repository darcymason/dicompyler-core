import unittest
import numpy
from dicompylercore.dvhcalc import get_dvh
from .util import fake_rtdose, fake_ss


class TestDVHCalcDecubitus(unittest.TestCase):
    """Unit tests for DVH calculation in decubitus orientations."""

    def setUp(self):
        self.ss = fake_ss()
        self.dose = fake_rtdose()

    def test_nondecub(self):
        """Test that DVH is calculated correctly for standard orientation."""
        self.dose.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dvh = get_dvh(self.ss, self.dose, 1)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        expected_counts = [0]*13 + [2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                                    2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_HF_decubitus_left(self):
        """Test DVH for head-first decubitus left orientation."""
        # Keep same dose grid as std orientation but pixel-spacing in X, Y same
        # For this case, use iop=[0, -1, 0, 1, 0, 0] Head first decubitus left
        # Then X = r * dr + ipp[0]
        #  and Y = -c * dc + ipp[1]
        # (https://nipy.org/nibabel/dicom/dicom_orientation.html
        # #dicom-affine-formula)
        # Change ipp y of y to new max of 19 for similar y range
        # Below show contours box of (3, 14.5) - (7, 17.5) on dose grid
        #       Y=19 18  17                  12
        # X=2   [10, 10, 10, 13, 14, 15, 16, 17],
        #               |-----------|
        #   4   [10, 10, 10, 13, 14, 15, 16, 17]
        #   6   [10, 10, 10, 13, 14, 15, 16, 17]
        #               |-----------|
        #   8   [13, 13, 13, 16, 17, 18, 19, 20]
        #  10   [14, 14, 14, 17, 18, 19, 20, 21]
        #  12   [15, 15, 15, 18, 19, 20, 21, 22]
        #  14   [16, 16, 16, 19, 20, 21, 22, 23]]

        #       Y=19 18  17                  12
        # X=2   [20, 20, 20, 23, 24, 25, 26, 27]
        #               |-----------|
        #   4   [20, 20, 20, 23, 24, 25, 26, 27]
        #   6   [20, 20, 20, 23, 24, 25, 26, 27]
        #               |-----------|
        #   8   [23, 23, 23, 26, 27, 28, 29, 30]
        #  10   [24, 24, 24, 27, 28, 29, 30, 31]
        #  12   [25, 25, 25, 28, 29, 30, 31, 32]
        #  14   [...]

        #       Y=19 18  17                  12
        # X=2   [30, 30, 30, 33, 34, 35, 36, 37]
        #               |-----------|
        #   4   [30, 30, 30, 33, 34, 35, 36, 37]
        #   6   [30, 30, 30, 33, 34, 35, 36, 37]
        #               |-----------|
        #   8   [33, 33, 33, 36, 37, 38, 39, 40]
        #   10  [34, 34, 34, 37, 38, 39, 40, 41]
        #   12  [35, 35, 35, 38, 39, 40, 41, 42]
        # X=14  [36, 36, 36, 39, 40, 41, 42, 43]

        #                          10       13 14                20
        expected_counts = [0]*10 + [2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0,
                                    2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2]
        #                          23 24                30       33 34
        self.dose.ImagePositionPatient = [2, 19, -20]  # X Y Z top left
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        dvh = get_dvh(self.ss, self.dose, 1)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_HF_decubitus_left_structure_extents(self):
        """Test DVH for HF decubitus Lt orientation structure_extents used."""
        # Repeat test_HF_decubitus_left but with use_structure_extents
        #                          10       13 14                20
        expected_counts = [0]*10 + [2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0,
                                    2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2]
        #                          23 24                30       33 34
        self.dose.ImagePositionPatient = [2, 19, -20]  # X Y Z top left
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        dvh = get_dvh(self.ss, self.dose, 1, use_structure_extents=True)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_HF_decubitus_right(self):
        """Test DVH for head-first decubitus right orientation."""
        # Keep same dose grid as std orientation

        self.dose.ImageOrientationPatient = [0, 1, 0, -1, 0, 0]
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        # original ipp = [2, 12, -20]
        # Then X = -r * dr + ipp[0], X decreases down the rows
        #  and Y = c * dc + ipp[1], Y increases across cols
        # (https://nipy.org/nibabel/dicom/dicom_orientation.html
        # #dicom-affine-formula)
        # Change ipp y of X to new max of 14 for similar y range
        self.dose.ImagePositionPatient = [14, 12, -20]  # X Y Z top left
        # Below show contours box of (3, 14.5) - (7, 17.5) on dose grid
        #       Y=12 13  14  15  16  17  18  19
        # X=14  [10, 10, 10, 13, 14, 15, 16, 17],
        #   12  [10, 10, 10, 13, 14, 15, 16, 17]
        #   10  [10, 10, 10, 13, 14, 15, 16, 17]
        #    8  [13, 13, 13, 16, 17, 18, 19, 20]
        #                   | ----------|
        #    6  [14, 14, 14, 17, 18, 19, 20, 21]
        #    4  [15, 15, 15, 18, 19, 20, 21, 22]
        #                   | ----------|
        #    2  [16, 16, 16, 19, 20, 21, 22, 23]]

        #       Y=12 13  14                  19
        # X=14  [20, 20, 20, 23, 24, 25, 26, 27]
        #   12  [20, 20, 20, 23, 24, 25, 26, 27]
        #   10  [20, 20, 20, 23, 24, 25, 26, 27]
        #    8  [23, 23, 23, 26, 27, 28, 29, 30]
        #                   | ----------|
        #    6  [24, 24, 24, 27, 28, 29, 30, 31]
        #    4  [25, 25, 25, 28, 29, 30, 31, 32]
        #                   | ----------|
        #    2  [...]

        #       Y=12 13  14                  19
        # X=14  [30, 30, 30, 33, 34, 35, 36, 37]
        #   12  [30, 30, 30, 33, 34, 35, 36, 37]
        #   10  [30, 30, 30, 33, 34, 35, 36, 37]
        #    8  [33, 33, 33, 36, 37, 38, 39, 40]
        #                   | ----------|
        #    6  [34, 34, 34, 37, 38, 39, 40, 41]
        #    4  [35, 35, 35, 38, 39, 40, 41, 42]
        #                   | ----------|
        # X= 2  [36, 36, 36, 39, 40, 41, 42, 43]

        #                           17       20
        expected_counts = [0]*17 + [1, 2, 2, 1, 0, 0, 0, 0, 0, 0,
                                    1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1]
        #                          27 28 29 30                   37
        dvh = get_dvh(self.ss, self.dose, 1)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_FF_decubitus_right(self):
        """Test DVH for feet-first decubitus right orientation."""
        self.dose.ImageOrientationPatient = [0, -1, 0, -1, 0, 0]
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        # original ipp = [2, 12, -20]
        # Then X = -r * dr + ipp[0], X decreases down the rows
        #  and Y = -c * dc + ipp[1], Y decreases across cols
        # (https://nipy.org/nibabel/dicom/dicom_orientation.html
        # #dicom-affine-formula)
        self.dose.ImagePositionPatient = [14, 19, 20]  # X Y Z top left
        # Below show contours box of (3, 14.5) - (7, 17.5) on dose grid
        #       Y=19 18  17  16  15  14  13  12
        # X=14  [10, 10, 10, 13, 14, 15, 16, 17],
        #   12  [10, 10, 10, 13, 14, 15, 16, 17]
        #   10  [10, 10, 10, 13, 14, 15, 16, 17]
        #    8  [13, 13, 13, 16, 17, 18, 19, 20]
        #               | ----------|
        #    6  [14, 14, 14, 17, 18, 19, 20, 21]
        #    4  [15, 15, 15, 18, 19, 20, 21, 22]
        #               | ----------|
        #    2  [16, 16, 16, 19, 20, 21, 22, 23]]

        #       Y=19 18  17  16  15  14  13  12
        # X=14  [20, 20, 20, 23, 24, 25, 26, 27]
        #   12  [20, 20, 20, 23, 24, 25, 26, 27]
        #   10  [20, 20, 20, 23, 24, 25, 26, 27]
        #    8  [23, 23, 23, 26, 27, 28, 29, 30]
        #               | ----------|
        #    6  [24, 24, 24, 27, 28, 29, 30, 31]
        #    4  [25, 25, 25, 28, 29, 30, 31, 32]
        #               | ----------|
        #    2  [...]

        #       Y=19 18  17  16  15  14  13  12
        # X=14  [30, 30, 30, 33, 34, 35, 36, 37]
        #   12  [30, 30, 30, 33, 34, 35, 36, 37]
        #   10  [30, 30, 30, 33, 34, 35, 36, 37]
        #    8  [33, 33, 33, 36, 37, 38, 39, 40]
        #               | ----------|
        #    6  [34, 34, 34, 37, 38, 39, 40, 41]
        #    4  [35, 35, 35, 38, 39, 40, 41, 42]
        #               | ----------|
        # X= 2  [36, 36, 36, 39, 40, 41, 42, 43]

        #                          14 15 16       19             24
        expected_counts = [0]*14 + [1, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                                    2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 2, 1]
        #                                            34
        dvh = get_dvh(self.ss, self.dose, 1)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_FF_decubitus_right_structure_extents(self):
        """Test DVH for FF decubitus Rt orientation using structure extents."""
        self.dose.ImageOrientationPatient = [0, -1, 0, -1, 0, 0]
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        self.dose.ImagePositionPatient = [14, 19, 20]  # X Y Z top left
        # see grid from test_FF_decubitus_right
        #                          14 15 16       19             24
        expected_counts = [0]*14 + [1, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0,
                                    1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 1, 2, 1]
        #                                               34
        dvh = get_dvh(self.ss, self.dose, 1, use_structure_extents=True)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_FF_decubitus_left(self):
        """Test DVH for feet-first decubitus left orientation."""
        self.dose.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns
        # original ipp = [2, 12, -20]
        # Then X = r * dr + ipp[0], X increases down the rows
        #  and Y = c * dc + ipp[1], Y increases across cols
        # (https://nipy.org/nibabel/dicom/dicom_orientation.html
        # #dicom-affine-formula)

        # In this test, we also shift Z so three structure planes use the
        #    first three dose planes rather than the middle three,
        #    just to ensure asymmetry in z direction is checked.
        #    Note, planes should really be reversed in pixel array, but doesn't
        #    matter since contour is identical on each slice.
        self.dose.ImagePositionPatient = [2, 12, 10]  # X Y Z top left
        # Below show contours box of (3, 14.5) - (7, 17.5) on dose grid
        #      Y=12  13  14  15  16  17      19
        # X=2   [ 0,  0,  0,  3,  4,  5,  6,  7],
        #                   |-----------|
        #   4   [ 0,  0,  0,  3,  4,  5,  6,  7]
        #   6   [ 0,  0,  0,  3,  4,  5,  6,  7]
        #                   |-----------|
        #   8   [ 3,  3,  3,  6,  7,  8,  9, 10]
        #  10   [ 4,  4,  4,  7,  8,  9, 10, 11]
        #  12   [ 5,  5,  5,  8,  9, 10, 11, 12]
        #  14   [ 6,  6,  6,  9, 10, 11, 12, 13]]

        #      Y=12  13  14                  19
        # X=2   [10, 10, 10, 13, 14, 15, 16, 17],
        #                   |-----------|
        #   4   [10, 10, 10, 13, 14, 15, 16, 17]
        #   6   [10, 10, 10, 13, 14, 15, 16, 17]
        #                   |-----------|
        #   8   [13, 13, 13, 16, 17, 18, 19, 20]
        #  10   [14, 14, 14, 17, 18, 19, 20, 21]
        #  12   [15, 15, 15, 18, 19, 20, 21, 22]
        #  14   [16, 16, 16, 19, 20, 21, 22, 23]]

        #      Y=12  13  14                  19
        # X=2   [20, 20, 20, 23, 24, 25, 26, 27]
        #                   |-----------|
        #   4   [20, 20, 20, 23, 24, 25, 26, 27]
        #   6   [20, 20, 20, 23, 24, 25, 26, 27]
        #                   |-----------|
        #   8   [23, 23, 23, 26, 27, 28, 29, 30]
        #  10   [24, 24, 24, 27, 28, 29, 30, 31]
        #  12   [25, 25, 25, 28, 29, 30, 31, 32]
        #  14   [...]

        #                          3
        expected_counts = [0]*3 + [2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                                   2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]
        #                         13                            23
        dvh = get_dvh(self.ss, self.dose, 1)
        diffl = dvh.differential
        # Counts are normalized to total, and to volume,
        # So undo that here for test dose grid.
        # 18=num dose voxels inside struct; 0.36=volume
        got_counts = diffl.counts * 18 / 0.36
        assert numpy.all(numpy.isclose(got_counts, expected_counts))

    def test_empty_dose_grid(self):
        # See #274, prior to fixes this raised IndexError from
        #  get_interpolated_dose() getting empty array from GetDoseGrid()
        # Use z value to force no dose grid at that value
        #  Otherwise make like decub example
        self.dose.ImagePositionPatient = [2, 19, -1020]  # X Y Z top left
        self.dose.PixelSpacing = [2.0, 1.0]  # between Rows, Columns

        # 1 = roi number
        dvh = get_dvh(self.ss, self.dose, 1, use_structure_extents=True)
        self.assertTrue('Empty DVH' in dvh.notes)

    def test_not_implemented_orientations(self):
        """Test unhandled orientations raise NotImplementedError."""
        self.dose.ImageOrientationPatient = [0.7071, 0.7071, 0, 1, 0, 0]
        with self.assertRaises(NotImplementedError):
            _ = get_dvh(self.ss, self.dose, 1)


if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
