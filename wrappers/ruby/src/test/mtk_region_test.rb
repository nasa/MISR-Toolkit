$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_region"
require "test/unit"

class MtkRegionTest < Test::Unit::TestCase

  def setup
    @mtk_region = MtkRegion.create_extent_meters(10, 20, 3000, 4000)
  end
  def test_basic
    assert_instance_of MtkRegion, @mtk_region
    assert_equal [10, 20], @mtk_region.center
    assert_equal [3000, 4000], @mtk_region.extent
  end
  def test_extent_degrees
    @mtk_region = MtkRegion.create_extent_degrees(10, 20, 1, 1)
    assert_equal [10, 20], @mtk_region.center
    assert_in_delta 111132.84, @mtk_region.extent[0], 0.01
    assert_in_delta 111132.84, @mtk_region.extent[1], 0.01
  end
  def test_extent_pixel
    @mtk_region = MtkRegion.create_extent_pixel(10, 20, 100, 30, 40)
    assert_instance_of MtkRegion, @mtk_region
    assert_equal [10, 20], @mtk_region.center
    assert_equal [3000, 4000], @mtk_region.extent
  end
  def test_extent_path
    @mtk_region = MtkRegion.create_path_block_range(180, 83..83)
    assert_in_delta 9.40, @mtk_region.center[0], 0.01
    assert_in_delta 20.98, @mtk_region.center[1], 0.01
    assert_in_delta 140525, @mtk_region.extent[0], 1
    assert_in_delta 562925, @mtk_region.extent[1], 1
  end
  def test_extent_create_corner
    @mtk_region = MtkRegion.create_corners(10.5, 19.5, 9.5, 20.5)
    assert_equal [10, 20], @mtk_region.center
    assert_in_delta 111319.55, @mtk_region.extent[0], 0.01
    assert_in_delta 111319.55, @mtk_region.extent[1], 0.01
  end
  def test_path_list
    assert_equal [180, 181, 182], @mtk_region.path_list
  end
  def test_block_range
    assert_equal 83..83, @mtk_region.block_range(180)
  end
end
