import MisrToolkit
import unittest

class TestMtkFile(unittest.TestCase):
    
    def setUp(self):
        self.mtk_file = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AGP_P036_F01_24.hdf')

    def testnew(self):
        self.assertRaises(IOError, MisrToolkit.MtkFile, 'abcd.hdf')

    def testlocal_granule_id(self):
        self.assertEqual(self.mtk_file.local_granule_id, 'MISR_AM1_AGP_P036_F01_24.hdf')

    def testblock(self):
        self.assertEqual(self.mtk_file.block, (48, 80))

    def testgrid(self):
        self.assertEqual(type(self.mtk_file.grid('Standard')),MisrToolkit.MtkGrid)
        self.assertRaises(NameError, self.mtk_file.grid, 'abcd')

    def testgrid_list(self):
        self.assertEqual(self.mtk_file.grid_list, ['Standard', 'Regional'])

    def testpath(self):
        self.assertEqual(self.mtk_file.path, 36)

    def testorbit(self):
        mtk_file = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf');
        self.assertEqual(mtk_file.orbit, 29058)

    def testfile_type(self):
        self.assertEqual(self.mtk_file.file_type,'AGP')
    
    def testfile_name(self):
        self.assertEqual(self.mtk_file.file_name, '../../../../Mtk_testdata/in/MISR_AM1_AGP_P036_F01_24.hdf')

    def testversion(self):
        self.assertEqual(self.mtk_file.version, 'F01_24')

    def testcore_metadata_get(self):
        self.assertEqual(self.mtk_file.core_metadata_get('LOCALGRANULEID'),
                         'MISR_AM1_AGP_P036_F01_24.hdf')

    def testcore_metadata_list(self):
        self.assertEqual(self.mtk_file.core_metadata_list,['LOCALGRANULEID', 
          'PRODUCTIONDATETIME', 'LOCALVERSIONID', 'PGEVERSION', 'VERSIONID', 
          'SHORTNAME', 'GPOLYGONCONTAINER', 'GRINGPOINTLONGITUDE', 
          'GRINGPOINTLATITUDE', 'GRINGPOINTSEQUENCENO', 'EXCLUSIONGRINGFLAG', 
          'RANGEENDINGDATE', 'RANGEENDINGTIME', 'RANGEBEGINNINGDATE', 
          'RANGEBEGINNINGTIME', 'ADDITIONALATTRIBUTESCONTAINER', 
          'ADDITIONALATTRIBUTENAME', 'PARAMETERVALUE', 
          'ADDITIONALATTRIBUTESCONTAINER', 'ADDITIONALATTRIBUTENAME', 
          'PARAMETERVALUE', 'ADDITIONALATTRIBUTESCONTAINER', 
          'ADDITIONALATTRIBUTENAME', 'PARAMETERVALUE'])

    def testattr_get(self):
        self.assertEqual(self.mtk_file.attr_get('Start_block'),48)
        self.assertEqual(self.mtk_file.attr_get('Translation.land_water_id'),
                         [0, 1, 2, 3, 4, 3, 0])

    def testattr_list(self):
        self.assertEqual(self.mtk_file.attr_list,['HDFEOSVersion',
          'StructMetadata.0', 'Translation.number_id',
          'Translation.land_water_id', 'Translation.dark_water_mask',
          'Path_number', 'AGP_version_id', 'DID_version_id', 'Number_blocks',
          'Ocean_blocks_size', 'Ocean_blocks.count', 'Ocean_blocks.numbers',
          'SOM_parameters.som_ellipsoid.a', 'SOM_parameters.som_ellipsoid.e2',
          'SOM_parameters.som_orbit.aprime', 'SOM_parameters.som_orbit.eprime',
          'SOM_parameters.som_orbit.gama', 'SOM_parameters.som_orbit.nrev',
          'SOM_parameters.som_orbit.ro', 'SOM_parameters.som_orbit.i',
          'SOM_parameters.som_orbit.P2P1', 'SOM_parameters.som_orbit.lambda0',
          'Origin_block.ulc.x', 'Origin_block.ulc.y', 'Origin_block.lrc.x',
          'Origin_block.lrc.y', 'Start_block', 'End block', 'coremetadata',
          'SubsetMetadata'])

    def testblock_metadata_list(self):
        self.assertEqual(self.mtk_file.block_metadata_list,['PerBlockMetadataCommon'])

    def testblock_metadata_field_list(self):
        self.assertEqual(self.mtk_file.block_metadata_field_list('PerBlockMetadataCommon'),
                         ['Block_number', 'Ocean_flag', 'Block_coor_ulc_som_meter.x',
                          'Block_coor_ulc_som_meter.y', 'Block_coor_lrc_som_meter.x',
                          'Block_coor_lrc_som_meter.y', 'Data_flag'])
        self.assertRaises(NameError, self.mtk_file.block_metadata_field_list, 'abcd')

    def testblock_metadata_field_read(self):
        self.assertEqual(self.mtk_file.block_metadata_field_read('PerBlockMetadataCommon', 'Block_number'),
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 49, 50, 51, 52, 53,
                          54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                          75, 76, 77, 78, 79, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.mtk_file.block_metadata_field_read('PerBlockMetadataCommon', 'Block_coor_lrc_som_meter.x')[47],14219150.0)
        f = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_AS_LAND_P037_O029058_F06_0017.hdf')
        self.assertEqual(f.block_metadata_field_read('PerBlockMetadataTime','BlockCenterTime')[5],'2005-06-04T18:00:40.274433Z')
        self.assertRaises(NameError, self.mtk_file.block_metadata_field_read, 'abcd', 'Block_number')
        self.assertRaises(NameError, self.mtk_file.block_metadata_field_read, 'PerBlockMetadataCommon', 'abcd')
        f = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf')
        self.assertEqual(f.block_metadata_field_read('PerBlockMetadataRad', 'transform.ref_time')[0],['2005-06-04T17:58:13.127920Z', '2005-06-04T17:58:13.127920Z'])
        self.assertEqual(f.block_metadata_field_read('PerBlockMetadataRad', 'transform.start_line')[2], [1024, 1280])

    def testtime_metadata_read(self):
        f = MisrToolkit.MtkFile('../../../../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf')
        self.assertEqual(type(f.time_metadata_read()),MisrToolkit.MtkTimeMetaData)
        
if __name__ == '__main__':
    unittest.main()
