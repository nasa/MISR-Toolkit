$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "hdf_file"
require "hdf_vdata"
require "test/unit"

class HdfVdataTest < Test::Unit::TestCase
  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    d = ENV["MTKTESTHOME"] + "/in/"
    @fname = d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @hdf_file = HdfFile.new(@fname)
    @vdata = @hdf_file.vdata("PerBlockMetadataCommon")
  end
  def test_basic
    assert_equal(HdfVdata, @vdata.class)
  end
  def test_field_list
    assert_equal ["Block_number",
      "Ocean_flag",
      "Block_coor_ulc_som_meter.x",
      "Block_coor_ulc_som_meter.y",
      "Block_coor_lrc_som_meter.x",
      "Block_coor_lrc_som_meter.y",
      "Data_flag" ], @vdata.field_list
  end
  def test_read
    start_block = 48
    end_block = 80
    expected = Array.new(180) do |i| 
      b = i + 1
      (b < start_block || b > end_block ? 0 : b)
    end
    assert_equal expected, @vdata.read("Block_number")
    assert_equal expected[50...180], @vdata.read("Block_number", 50)
    assert_equal expected[50...70], @vdata.read("Block_number", 50, 20)
  end
  def test_read_bad
    assert_raise(RuntimeError) {@vdata.read("Bad field") }
    assert_raise(RuntimeError) {@vdata.read("Block number", -1) }
    assert_raise(RuntimeError) {@vdata.read("Block number", 1000) }
    assert_raise(RuntimeError) {@vdata.read("Block number", 0, 100) }
  end
  def test_size
    assert_equal 180, @vdata.size
  end
end
