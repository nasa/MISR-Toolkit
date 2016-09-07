import MisrToolkit
import unittest
import numpy

class TestMtkTimeMetaData(unittest.TestCase):

    def setUp(self):
        self.time_metadata = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf').time_metadata_read()

    def testpixel_time(self):
        self.assertEqual(self.time_metadata.pixel_time(10153687.5, 738787.5),'2005-06-04T18:06:07.656501Z')
        self.assertEqual(self.time_metadata.pixel_time(10153687.5, 1020387.5),'2005-06-04T18:06:07.522389Z')
        self.assertEqual(self.time_metadata.pixel_time(10224087.5, 738787.5),'2005-06-04T18:06:18.002869Z')
        self.assertEqual(self.time_metadata.pixel_time(10224087.5, 1020387.5),'2005-06-04T18:06:17.839556Z')

    def testpath(self):
        self.assertEqual(self.time_metadata.path,37)
        
    def teststart_block(self):
        self.assertEqual(self.time_metadata.start_block,1)

    def testend_block(self):
        self.assertEqual(self.time_metadata.end_block,140)
        
    def testcamera(self):
        self.assertEqual(self.time_metadata.camera,'AA')

    def testnumber_transform(self):
        self.assertEqual(self.time_metadata.number_transform[100],2)

    def testref_time(self):
        self.assertEqual(self.time_metadata.ref_time[100],['2005-06-04T17:58:13.127920Z','2005-06-04T17:58:13.127920Z'])

    def teststart_line(self):
        self.assertEqual(self.time_metadata.start_line[100][0],50688)
        self.assertEqual(self.time_metadata.start_line[100][1],50944)

    def testnumber_line(self):
        self.assertEqual(self.time_metadata.number_line[100][0],256)
        self.assertEqual(self.time_metadata.number_line[100][1],256)

    def testcoeff_line(self):
        coeff_line = numpy.array([[5.2314916E+04, 5.2567776E+04],
                                  [9.8764910E-01, 9.8778724E-01],
                                  [-6.4932617E-02, -6.4070067E-02],
                                  [1.0238983E-05, 1.0254441E-05],
                                  [3.3763384E-06, 3.4716382E-06],
                                  [5.3088579E-11, 5.5760239E-11]])
        for i in range(6):
            for j in range(2):
                self.assertAlmostEqual(self.time_metadata.coeff_line[100][i][j], coeff_line[i][j], 3)

    def testsom_ctr_x(self):
        som_ctr_x = numpy.array([5.0816000E+04, 5.1072000E+04])
        for i in range(2):
            self.assertAlmostEqual(self.time_metadata.som_ctr_x[100][i], som_ctr_x[i], 3)
        
    def testsom_ctr_y(self):
        som_ctr_y = numpy.array([1.0240000E+03, 1.0240000E+03])
        for i in range(2):
            self.assertAlmostEqual(self.time_metadata.som_ctr_y[100][i], som_ctr_y[i], 3)

if __name__ == '__main__':
    unittest.main()
