import unittest

import mtkdataplane_test
import mtkfield_test
import mtkfile_test
import mtkgrid_test
import mtkmapinfo_test
import mtkregion_test
import mtktimemetadata_test

suite1 = unittest.makeSuite(mtkdataplane_test.TestMtkDataPlane)
suite2 = unittest.makeSuite(mtkfield_test.TestMtkField)
suite3 = unittest.makeSuite(mtkfile_test.TestMtkFile)
suite4 = unittest.makeSuite(mtkgrid_test.TestMtkGrid)
suite5 = unittest.makeSuite(mtkmapinfo_test.TestMtkMapInfo)
suite6 = unittest.makeSuite(mtkregion_test.TestMtkRegion)
suite6 = unittest.makeSuite(mtktimemetadata_test.TestMtkTimeMetaData)


MisrToolkitTestSuite = unittest.TestSuite((suite1, suite2, suite3, suite4, suite5, suite6))
                                     
if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(MisrToolkitTestSuite)
