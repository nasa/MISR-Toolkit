$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_file"
require "test/unit"

class MtkCoordinateTest < Test::Unit::TestCase

  def test_block_to_lat_lon
    lat, lon = MtkCoordinate.block_to_lat_lon(23, 1100, 10, 0, 0)
    assert_in_delta lat, 76.833, 0.001
    assert_in_delta lon, 63.097, 0.001
  end
end
