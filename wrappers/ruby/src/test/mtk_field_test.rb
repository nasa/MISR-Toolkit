$:.unshift File.join(File.dirname(__FILE__), "..", "lib")
require "mtk_file"
require "test/unit"

class MtkFieldTest < Test::Unit::TestCase
  def setup
    raise "Need to set MTKTESTHOME to point to directory with test data to run unit tests" unless(ENV["MTKTESTHOME"])
    @d = ENV["MTKTESTHOME"] + "/in/"
    @fname = @d + "MISR_AM1_AGP_P038_F01_24.hdf"
    @mtk_file = MtkFile.new(@fname)
    @mtk_grid = @mtk_file.grid("Standard")
    @mtk_field = @mtk_grid.field("AveSceneElev")
    @mtk_region = MtkRegion.create_path_block_range(38, 50..51)
  end
  def test_bad_field
    assert_raise(RuntimeError) {@mtk_grid.field("bad field")}
  end
  def test_data_type
    assert_equal MtkField::INT16, @mtk_field.data_type
  end
  def test_no_fill_value
    assert_raise(RuntimeError) {@mtk_field.fill_value}
  end
  def test_fill_value
    fname2 = @d + "MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf"
    field2 = MtkFile.new(fname2).grid("BlueBand").field("Blue Radiance/RDQI")
    assert_equal 65515, field2.fill_value
  end
  def test_wrong_extra_dim
    fname2 = @d + "MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
    assert_raise(RuntimeError) do
      MtkFile.new(fname2).grid("RegParamsAer").field("RegMeanSpectralOptDepth")
    end
  end
  def test_bad_extra_dim
    fname2 = @d + "MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
    assert_raise(RuntimeError) do
      MtkFile.new(fname2).grid("RegParamsAer").
        field("RegMeanSpectralOptDepth", -1)
    end
    assert_raise(RuntimeError) do
      MtkFile.new(fname2).grid("RegParamsAer").
        field("RegMeanSpectralOptDepth", 4)
    end
  end
  def test_extra_dim
    fname2 = @d + "MISR_AM1_AS_AEROSOL_P037_O029058_F09_0017.hdf"
    field2 = MtkFile.new(fname2).grid("RegParamsAer").
      field("RegMeanSpectralOptDepth", 1)
    mtk_region2 = MtkRegion.create_path_block_range(37, 50..51)
    data = field2.read(mtk_region2).data
    assert_equal [16, 32], data.shape
  end
  def test_read
    data = @mtk_field.read(@mtk_region).data
    assert_equal [256, 512], data.shape

# Print out values, used for debugging. Normally turned off
    if(false)
      print "\n[["
      (123...133).each do |i|
#      (0...10).each do |i|
        (0...9).each do |j|
          print data[i, j]
          if(j != 8)
            print ", "
          else
            if(i != 9 && i != 132)
              print "],\n["
            else
              print "]]\n"
            end
          end
        end
      end
    end

# These value were read from the input data using HDF scan
    expected =
[[779, 788, 805, 815, 825, 828, 826, 818, 826],
[776, 784, 797, 814, 821, 819, 816, 815, 826],
[775, 780, 792, 808, 811, 808, 809, 814, 820],
[776, 778, 787, 798, 801, 800, 804, 811, 815],
[779, 775, 779, 787, 791, 794, 802, 810, 816],
[785, 775, 775, 779, 783, 790, 802, 809, 813],
[791, 777, 774, 775, 777, 785, 798, 805, 807],
[796, 780, 774, 773, 775, 780, 791, 800, 803],
[802, 782, 775, 773, 773, 775, 785, 795, 799],
[807, 785, 776, 777, 771, 773, 783, 791, 794]]
    # NArray is in column major form, which is why the transpose is needed
    expected = NArray.to_na(expected).transpose(1, 0)
    assert_equal 0, expected.ne(data[0...10,0...9]).count_true

# Now, check as we cross the block boundary.
    expected =
[[769, 774, 780, 785, 793, 797, 798, 800, 800],
[787, 776, 771, 771, 785, 795, 797, 798, 800],
[791, 788, 783, 772, 779, 791, 793, 794, 797],
[793, 791, 789, 777, 777, 786, 790, 790, 792],
[794, 792, 790, 781, 774, 781, 787, 788, 787],
[794, 792, 791, 784, 774, 777, 785, 785, 784],
[794, 793, 791, 785, 775, 774, 782, 782, 780],
[794, 793, 790, 784, 776, 772, 777, 775, 772],
[795, 794, 792, 786, 779, 770, 772, 775, 782],
[795, 795, 794, 790, 785, 779, 782, 787, 792]]
    expected = NArray.to_na(expected).transpose(1, 0)
    assert_equal 0, expected.ne(data[123...133,0...9]).count_true
  end
end
