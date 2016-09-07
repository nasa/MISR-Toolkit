require "mtk_data_plane"
require "mtk.so"

# This is used to read a field
class MtkField

# Field name
  attr_reader :field_name

# If a field has more than the three block, som x, som y dimensions, we
# specify the value for the extra dimensions (such as band) in this array
 attr_reader :extra_dim

# Grid that field belongs to
  attr_reader :grid

# :call-seq:
#   mtk_grid.field(field_name, extra_dim_1, extra_dim_2, ...) -> MtkField
#
# Create an object to read a field in a MtkGrid. If a field has more than 
# the three block, som x, som y dimensions, we specify the value for the 
# extra dimensions (such as band) in the argument list.
#
  def initialize(g, f, ed)
    @grid = g
    @extra_dim = ed
    @field_name = f
    raise "Field #{field_name} is not found in grid #{grid.grid_name} of file #{grid.file.file_name}" unless
      grid.field_list.include?(field_name)
    ds = grid.field_dim_size(field_name)
    raise "Extra dimensions passed to constructor of field #{field_name} must match size of extra dimensions" unless(ds.size ==@extra_dim.size)
    (0...(@extra_dim.size)).each do |i|
      raise "Extra dimension out of range for field #{field_name}" unless
        (@extra_dim[i] >= 0 && @extra_dim[i] < ds[i])
    end
  end

# :call-seq:
#   data_type -> Integer
#
# Return data type of field. This matchs one of the constants defined in
# MtkField (e.g, CHAR8).
#
  def data_type
    MtkField.field_to_datatype(grid.file.file_name, grid.grid_name, field_name)
  end

# :call-seq:
#   fill_value -> Number or Float
#
# Return fill value of field.
#
  def fill_value
    MtkField.field_to_fillvalue(grid.file.file_name, grid.grid_name, field_name)
  end

# :call-seq:
#   read(region) -> MtkDataPlane
#
# Read the field for the given MtkRegion, returning a MtkDataPlane.
#
  def read(region)
    f = field_name
    f = f + "[" + extra_dim.join("][") + "]" if(extra_dim.size > 0)
    MtkField.field_read(grid.file.file_name, grid.grid_name, 
                        f, region)
  end
end
