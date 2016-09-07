import MisrToolkit
import unittest
import numpy

class TestMtkDataPlane(unittest.TestCase):

    def setUp(self):
        self.region = MisrToolkit.MtkRegion(37,40,45)
        self.file = '../../../../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf'
        self.grid = 'Standard'
        self.field = 'AveSceneElev'
        self.mtk_dataplane = MisrToolkit.MtkFile(self.file).grid(self.grid).field(self.field).read(self.region)

    def testdata(self):
        self.assertEqual(type(self.mtk_dataplane.data()),numpy.ndarray)
        
    def testmapinfo(self):
        self.assertEqual(type(self.mtk_dataplane.mapinfo()),MisrToolkit.MtkMapInfo)


if __name__ == '__main__':
    unittest.main()
