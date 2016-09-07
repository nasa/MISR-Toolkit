$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "hdf_file"
require "test/unit"

class HdfFileTest < Test::Unit::TestCase

  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    d = ENV["MTKTESTHOME"] + "/in/"
    @fname = d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @arp_fname = d + "MISR_AM1_ARP_INFLTCAL_T044_F02_0010.hdf"
    @hdf_file = HdfFile.new(@fname)
  end
  def test_basic
    assert_equal(HdfFile, @hdf_file.class)
  end
  def test_bad_file
    assert_raise(RuntimeError) {HdfFile.new("bad_file")}
  end
  def test_grid_list
    assert_equal ["Standard", "Regional"], @hdf_file.grid_list
  end
  def test_grid
    @g = @hdf_file.grid("Standard")
    assert_equal(HdfGrid, @g.class)
  end
  def test_bad_grid
    assert_raise(RuntimeError) {@hdf_file.grid("Bad grid")}
  end
  def test_attributes
    @hdf_file = HdfFile.new(@arp_fname)
    assert_equal @hdf_file.attributes["TITLE"], 
    "MISR In-Flight Calibration ARP File"
  end
  def test_sds_list
    @hdf_file = HdfFile.new(@arp_fname)
    assert_equal @hdf_file.sds_list.sort.first, "abs_rad_unc_sys"
  end
  def test_sds_size
    @hdf_file = HdfFile.new(@arp_fname)
    assert_equal @hdf_file.sds_size("abs_rad_unc_sys"), [9, 4, 15]
  end
  def test_sds_size_bad_sds
    @hdf_file = HdfFile.new(@arp_fname)
    assert_raise(RuntimeError) {@hdf_file.sds_size("Bad sds")}
  end
  def test_sds_type
    @hdf_file = HdfFile.new(@arp_fname)
    assert_equal @hdf_file.sds_type("abs_rad_unc_sys"), 5
  end
  def test_sds_type_bad_sds
    @hdf_file = HdfFile.new(@arp_fname)
    assert_raise(RuntimeError) {@hdf_file.sds_type("Bad sds")}
  end
  def test_sds_read
    @hdf_file = HdfFile.new(@arp_fname)
    assert_equal @hdf_file.sds_read("abs_rad_unc_sys").shape, [9, 4, 15]
    assert_in_delta @hdf_file.sds_read("abs_rad_unc_sys")[0, 0, 0], 2.385456, 1e-6
  end
  def test_sds_read_bad_sds
    @hdf_file = HdfFile.new(@arp_fname)
    assert_raise(RuntimeError) {@hdf_file.sds_read("Bad sds")}
  end
  def test_vdata_list
    assert_equal 32, @hdf_file.vdata_list.size
  end
  def test_vdata
    @v = @hdf_file.vdata("PerBlockMetadataCommon")
    assert_equal(HdfVdata, @v.class)
  end
  def test_bad_vdata
    assert_raise(RuntimeError) {@hdf_file.vdata("Bad vdata")}
  end
end
