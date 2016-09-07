import MisrToolkit
import unittest

class TestMtkRegion(unittest.TestCase):
    
    def setUp(self):
        self.region = MisrToolkit.MtkRegion(39,50,60)

    def testblock_range(self):
        self.assertEqual(self.region.block_range(34), (54, 61))
        self.assertEqual(self.region.block_range(35), (50, 61))
        self.assertEqual(self.region.block_range(36), (50, 61))
        self.assertEqual(self.region.block_range(37), (50, 61))
        self.assertEqual(self.region.block_range(38), (50, 61))
        self.assertEqual(self.region.block_range(39), (50, 60))
        self.assertEqual(self.region.block_range(40), (49, 60))
        self.assertEqual(self.region.block_range(41), (49, 60))
        self.assertEqual(self.region.block_range(42), (49, 60))
        self.assertEqual(self.region.block_range(43), (49, 60))
        self.assertEqual(self.region.block_range(44), (49, 56))

    def testcenter(self):
        self.assertEqual(self.region.center, (44.327741112333754, -112.0139568316069))

    def testextent(self):
        self.assertEqual(self.region.extent, (774262.5, 343062.5))

    def testpath_list(self):
        self.assertEqual(self.region.path_list, [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])

    def testinit(self):
        self.assert_(MisrToolkit.MtkRegion(39,50,60))
        self.assert_(MisrToolkit.MtkRegion(40.0, -120.0, 30.0, -110.0))
        self.assert_(MisrToolkit.MtkRegion(35.0, -115.0, 1.5, 2.0, "deg"))
        self.assert_(MisrToolkit.MtkRegion(35.0, -115.0, 5000.0, 8000.0, "m"))
        self.assert_(MisrToolkit.MtkRegion(35.0, -115.0, 2.2, 1.1, "km"))
        self.assert_(MisrToolkit.MtkRegion(35.0, -115.0, 45.0, 100.0, "275m"))
        self.assert_(MisrToolkit.MtkRegion(35.0, -115.0, 35.0, 25.0, "1.1km"))

    def snap_to_grid(self):
        r = MisrToolkit.MtkRegion(37,1,1)
        mapinfo = r.snap_to_grid(37,1100)
        self.assertEqual((mapinfo.som.lrc.x - mapinfo.som.ulc.x) / mapinfo.resolution, mapinfo.nline-1)
        self.assertEqual((mapinfo.som.lrc.y - mapinfo.som.ulc.y) / mapinfo.resolution, mapinfo.nsample-1)

if __name__ == '__main__':
    unittest.main()
