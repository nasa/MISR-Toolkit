import MisrToolkit
import unittest

class TestMtkGrid(unittest.TestCase):
    
    def setUp(self):
        self.mtk_grid = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AGP_P036_F01_24.hdf').grid('Standard')

    def testfield(self):
        self.assertEqual(type(self.mtk_grid.field('AveSurfNormZenAng')), MisrToolkit.MtkField)
        self.assertRaises(NameError, self.mtk_grid.field, 'abcd')
        self.assertRaises(NameError, self.mtk_grid.field, 'AveSurfNormZenAng[0]')
        grid = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf').grid('SubregParamsLnd')
        self.assertEqual(type(grid.field('LAIDelta1[1]')), MisrToolkit.MtkField)
        self.assertRaises(NameError, grid.field, 'LAIDelta1')
        self.assertRaises(NameError, grid.field, 'LAIDelta1[200]')
        self.assertRaises(NameError, grid.field, 'LAIDelta1[0][1]')
        self.assertRaises(NameError, grid.field, 'badfield[0]')

    def testfield_dims(self):
        self.assertEqual(self.mtk_grid.field_dims('AveSceneElev'),[])
        grid = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf').grid('RegParamsAer')
        self.assertEqual(grid.field_dims('RegBestEstimateSpectralOptDepth'),[('NBandDim',4)])
        self.assertRaises(NameError, grid.field_dims, 'abcd')

    def testfield_list(self):
        self.assertEqual(self.mtk_grid.field_list, ['AveSceneElev',
            'StdDevSceneElev', 'StdDevSceneElevRelSlp', 'PtElev',
            'GeoLatitude', 'GeoLongitude', 'SurfaceFeatureID',
            'AveSurfNormAzAng', 'AveSurfNormZenAng'])

    def testresolution(self):
        self.assertEqual(self.mtk_grid.resolution, 1100)

    def testgrid_name(self):
        self.assertEqual(self.mtk_grid.grid_name, 'Standard')

    def testattr_get(self):
        self.assertEqual(self.mtk_grid.attr_get('Block_size.resolution_x'),1100)

    def testattr_list(self):
        self.assertEqual(self.mtk_grid.attr_list, ['Block_size.resolution_x',
          'Block_size.resolution_y', 'Block_size.size_x', 'Block_size.size_y'])
            
if __name__ == '__main__':
    unittest.main()
