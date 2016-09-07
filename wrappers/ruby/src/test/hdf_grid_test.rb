$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "hdf_file"
require "hdf_grid"
require "test/unit"

class HdfGridTest < Test::Unit::TestCase
  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    d = ENV["MTKTESTHOME"] + "/in/"
    @fname = d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @hdf_file = HdfFile.new(@fname)
    @grid = @hdf_file.grid("Standard")
  end
  def test_basic
    assert_equal(HdfGrid, @grid.class)
  end
  def test_field_list
    assert_equal ["AveSceneElev",
      "StdDevSceneElev",
      "StdDevSceneElevRelSlp",
      "PtElev",
      "GeoLatitude",
      "GeoLongitude",
      "SurfaceFeatureID",
      "AveSurfNormAzAng",
      "AveSurfNormZenAng"], @grid.field_list
  end
  def test_field_rank
    assert_equal 3, @grid.field_rank("AveSceneElev")
  end
  def test_field_dimlist
    assert_equal @grid.field_dimlist("AveSceneElev"), ["SOMBlockDim", "XDim", "YDim"]
  end
  def test_bad_field_rank
    assert_raise(RuntimeError) {@grid.field_rank("bad field") }
  end
  def test_field_size
    assert_equal [180, 128, 512], @grid.field_size("AveSceneElev")
  end
  def test_bad_field_size
    assert_raise(RuntimeError) {@grid.field_size("bad field") }
  end
  def test_field_read
    assert_equal [1, 128, 512], 
    @grid.field_read("AveSceneElev", 
                     :start => [10, 0, 0], :edge => [1, true, true]).shape
  end
  def test_bad_field_read
    assert_raise(RuntimeError) {@grid.field_read("bad field") }
    assert_raise(RuntimeError) {@grid.field_read("AveSceneElev", 
                                                 :start => [-1, 0, 0],
                                                 :stride => [1, 1, 1],
                                                 :edge => [1, true, true]) }
  end
end
