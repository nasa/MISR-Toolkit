$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_file"
require "test/unit"

class MtkFileTest < Test::Unit::TestCase

  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    d = ENV["MTKTESTHOME"] + "/in/"
    @fname = d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @mtk_file = MtkFile.new(@fname)
  end
  def test_basic
    assert_equal @fname, @mtk_file.file_name
  end
  def test_path
    assert_equal 38, @mtk_file.path
  end
  def test_block
    assert_equal 48..80, @mtk_file.block
  end
  def test_grid_list
    assert_equal ["Standard", "Regional"], @mtk_file.grid_list
  end
  def test_local_granule_id
    assert_equal "MISR_AM1_AGP_P038_F01_24.hdf", @mtk_file.local_granule_id
  end
  def test_file_type
    assert_equal MtkFile::AGP, @mtk_file.file_type
  end
  def test_version
    assert_equal "F01_24", @mtk_file.version
  end
end
