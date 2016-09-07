require "narray"
require "mtk.so"
require "mtk_region"
require "mtk_field"
require "mtk_grid"
require "mtk_data_plane"
require "mtk_coordinate"
require "hdf_file"
require "hdf_grid"
require "hdf_vdata"

# This is used to read a file, both for metadata and data.
#
class MtkFile
# File name being read
  attr_reader :file_name

# :call-seq:
#   MtkFile.new(Filename) -> MtkFile
#
# Create an object to read the given file.
#
  def initialize(f)
    @file_name = f
  end

# :call-seq:
#   path -> Integer
#
#  Return path for the given file
#
  def path
    MtkFile.file_to_path(file_name)
  end

# :call-seq:
#   block -> Range
#
# Return the block range for the given file
#
  def block
    MtkFile.file_to_block(file_name)
  end

# :call-seq:
#   grid_list -> Array
#
# Return an Array of Strings containing the grid names.
#
  def grid_list
    @grid_list if(@grid_list)
    @grid_list = MtkFile.file_to_grid_list(file_name)
  end

# :call-seq:
#   grid(grid_name) -> MtkGrid
#
# Return an MtkGrid to read the given grid.
#
  def grid(grid_name)
    MtkGrid.new(self, grid_name)
  end

# :call-seq:
#   local_granule_id -> String
#
# Return local granule ID
#
  def local_granule_id
    MtkFile.file_to_local_granule_id(file_name)
  end

# :call-seq:
#   file_type -> Integer
#
# Return file type. This is an Integer equal to one of the constants defined 
# in this class, e.g., MtkFile::AGP
#
  def file_type
    MtkFile.file_to_file_type(file_name)
  end

# :call-seq:
#   version -> String
#
# Return version, e.g., "F01_0001".
#
  def version
    MtkFile.file_to_version(file_name)
  end
end
