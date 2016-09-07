$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_file"
require "test/unit"
require "narray"

class MtkDataPlaneTest < Test::Unit::TestCase
  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    d = ENV["MTKTESTHOME"] + "/in/"
    @fname = d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @mtk_file = MtkFile.new(@fname)
    @mtk_grid = @mtk_file.grid("Standard")
    @mtk_field = @mtk_grid.field("AveSceneElev")
    @mtk_region = MtkRegion.create_path_block_range(34, 40..41)
    @mtk_data_plane = @mtk_field.read(@mtk_region)
  end
  def test_data
    assert_equal [256, 512], @mtk_data_plane.data.shape
  end
  def test_lat_lon
    lat, lon = @mtk_data_plane.lat_lon
    assert_equal [256, 512], lat.shape
    assert_equal [256, 512], lon.shape
  end
  def test_lat_lon_to_line_sample
    lat, lon = @mtk_data_plane.lat_lon
    lat_lon = NArray.to_na([lat, lon])
    lat_lon.reshape!(256 * 512, 2)
    line_sample = @mtk_data_plane.lat_lon_to_line_sample(lat_lon)
    line_sample.reshape!(256, 512, 2)
    assert_in_delta line_sample[10, 20, 0], 10, 1e-3
    assert_in_delta line_sample[10, 20, 1], 20, 1e-3
  end
end
