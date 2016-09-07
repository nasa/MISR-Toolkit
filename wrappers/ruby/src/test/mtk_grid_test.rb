$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_grid"
require "mtk_file"
require "test/unit"

class MtkGridTest < Test::Unit::TestCase
  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    @d = ENV["MTKTESTHOME"] + "/in/"
    @fname = @d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @mtk_file = MtkFile.new(@fname)
    @mtk_grid = @mtk_file.grid("Standard")
  end
  def test_bad_grid
    assert_raise(RuntimeError) {@mtk_file.grid("bad grid")}
  end
  def test_field_list
    assert_equal [
      "AveSceneElev",
      "StdDevSceneElev",
      "StdDevSceneElevRelSlp",
      "PtElev",
      "GeoLatitude",
      "GeoLongitude",
      "SurfaceFeatureID",
      "AveSurfNormAzAng",
      "AveSurfNormZenAng"
    ], @mtk_grid.field_list
  end
  def test_field_dim_list
    fname2 = @d + "MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
    dimlist = MtkFile.new(fname2).grid("RegParamsAer").
      field_dim_list("RegMeanSpectralOptDepth")
    assert_equal ["NBandDim"], dimlist
    assert_equal [], @mtk_grid.field_dim_list("GeoLongitude")
  end
  def test_field_dim_size
    fname2 = @d + "MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
    dimsize = MtkFile.new(fname2).grid("RegParamsAer").
      field_dim_size("RegMeanSpectralOptDepth")
    assert_equal [4], dimsize
    assert_equal [], @mtk_grid.field_dim_size("GeoLongitude")
  end
  def test_resolution
    assert_equal 1100, @mtk_grid.resolution
  end
end
