import MisrToolkit
import unittest

class TestMtkField(unittest.TestCase):

    def setUp(self):
        self.mtk_field = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf').grid('Standard').field('AveSceneElev')

    def testread(self):
        # Read using MtkRegion
        r = MisrToolkit.MtkRegion(35.0, -115.0, 110.0, 110.0, 'km')
        self.assertEqual(type(self.mtk_field.read(r)),MisrToolkit.MtkDataPlane)
        # Read using block range
        data = self.mtk_field.read(26,40)
        self.assertEqual(data[0][74][262],357)
        self.assertEqual(data[14][74][262],256)
        # Argument Checks
        self.assertRaises(TypeError, self.mtk_field.read)
        self.assertRaises(TypeError, self.mtk_field.read, 26)
        self.assertRaises(TypeError, self.mtk_field.read, 26, 'A')

    def testdata_type(self):
        self.assertEqual(self.mtk_field.data_type, 'int16')

    def testfill_value(self):
        self.assertEqual(self.mtk_field.fill_value, 0)

    def testfield_name(self):
        self.assertEqual(self.mtk_field.field_name, 'AveSceneElev')


if __name__ == '__main__':
    unittest.main()
