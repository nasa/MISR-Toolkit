require "mtk.so"

# This is used to read a grid
class MtkGrid

# Grid name
  attr_reader :grid_name 

# File that grid belongs to
  attr_reader :file 

# :call-seq:
#   mtk_file.grid(grid_name) -> MtkGrid
#
# Create an object to read a grid in a MtkFile
#
  def initialize(f, g)
    @file = f
    @grid_name = g
    raise "Grid #{grid_name} is not found in file #{file.file_name}" unless
      file.grid_list.include?(grid_name)
  end

# :call-seq:
#    field_list -> Array
#
# Return an Array of String containing the field names for this grid
#
  def field_list
    @field_list if(@field_list)
    @field_list = MtkGrid.grid_to_field_list(file.file_name, grid_name)
  end

# :call-seq:
#    field(field_name, extra_dim_1, extra_dim_2, ...) -> MtkField
#
# Return an object to read the given field. If a field has more than 
# the three block, som x, som y dimensions, we specify the value for the 
# extra dimensions (such as band) in the argument list.
#
  def field(field_name, *extra_dim)
    MtkField.new(self, field_name, extra_dim)
  end

# :call-seq:
#   field_dim_list(field_name) -> Array of String
#
# Return list of dimension names for the given field. Note this only includes
# the extra dimension, the normal YDim and XDim aren't included in this list.

  def field_dim_list(field_name)
    MtkGrid.field_to_dimlist(file.file_name, grid_name, field_name)[0]
  end

# :call-seq:
#   field_dim_size(field_name) -> Array of Integer
#
# Return list of dimension sizes for the given field. Note this only includes
# the extra dimension, the normal YDim and XDim aren't included in this list.

  def field_dim_size(field_name)
    MtkGrid.field_to_dimlist(file.file_name, grid_name, field_name)[1]
  end

# : call-seq:
#   resolution -> Integer
#
# Return resolution in meters of this grid.
#
  def resolution
    MtkGrid.grid_to_resolution(file.file_name, grid_name)
  end
end
