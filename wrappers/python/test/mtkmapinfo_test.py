import MisrToolkit
import unittest

class TestMtkMapInfo(unittest.TestCase):

    def setUp(self):
        self.region = MisrToolkit.MtkRegion(37,40,45)
        self.file = '../../../../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf'
        self.grid = 'Standard'
        self.field = 'AveSceneElev'
        self.mtk_mapinfo = MisrToolkit.MtkFile(self.file).grid(self.grid).field(self.field).read(self.region).mapinfo()

    def testlatlon_to_ls(self):
        line, samp = self.mtk_mapinfo.latlon_to_ls(63.605565, -105.04118)
        self.assertAlmostEqual(line, 22.9999771118, 5)
        self.assertAlmostEqual(samp, 32.0000343323, 5)

    def testls_to_latlon(self):
        lat, lon = self.mtk_mapinfo.ls_to_latlon(23, 32)
        self.assertAlmostEqual(lat, 63.6055648, 5)
        self.assertAlmostEqual(lon, -105.041180832, 5)

    def testls_to_somxy(self):
        self.assertEqual(self.mtk_mapinfo.ls_to_somxy(23, 32), (12977800.0, 492800.0))

    def testnline(self):
        self.assertEqual(self.mtk_mapinfo.nline,768)
        
    def testnsample(self):
        self.assertEqual(self.mtk_mapinfo.nsample,560)

    def testpath(self):
        self.assertEqual(self.mtk_mapinfo.path,37)

    def testpixelcenter(self):
        self.assertEqual(self.mtk_mapinfo.pixelcenter,True)

    def testpp(self):
        self.assertEqual(type(self.mtk_mapinfo.pp),MisrToolkit.MtkProjParam)

    def testresfactor(self):
        self.assertEqual(self.mtk_mapinfo.resfactor,4)

    def testresolution(self):
        self.assertEqual(self.mtk_mapinfo.resolution,1100)

    def testsom(self):
        self.assertEqual(type(self.mtk_mapinfo.som),MisrToolkit.MtkSomRegion)
    
    def testgeo(self):
        self.assertEqual(type(self.mtk_mapinfo.geo),MisrToolkit.MtkGeoRegion)

    def testsomxy_to_ls(self):
        self.assertEqual(self.mtk_mapinfo.somxy_to_ls(12977800.0, 492800.0), (23, 32))

    def teststart_block(self):
        self.assertEqual(self.mtk_mapinfo.start_block,40)
    
    def testend_block(self):
        self.assertEqual(self.mtk_mapinfo.end_block,45)
        
    def testcreate_latlon(self):
        r = MisrToolkit.MtkRegion(37, 35, 36)
        m = r.snap_to_grid(37, 1100)
        lat, lon = m.create_latlon()
        self.assertAlmostEqual(lat[63][79], 69.025150217462524, 5)
        self.assertAlmostEqual(lon[63][79], -98.336073888584707, 5)


if __name__ == '__main__':
    unittest.main()
