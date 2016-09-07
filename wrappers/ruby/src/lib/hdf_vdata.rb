require "mtk.so"
require "narray"

# This class is used  to read an HDF-EOS vdata. In addition to reading a field
# using read(name), we also define member functions for each field. This 
# is the lower case version, to a field like "Optical Depth Average" can be read
# by obj.optical_depth_average

class HdfVdata
# :call-seq:
#    field_list -> Array
#
# Return list of field names
#
  def field_list
    self.field_list_raw.split(',')
  end

# call-seq:
#   read("Field name", start = 0, sz = true) -> Array of values
#
# Read the given vdata field. Can optionally give the start and number
# of records to read, the default is to read them all. As a convenience,
# if sz is true rather than an Intger, we read everything from start.
#

  def read(field_name, start = 0, sz = true)
    sz = size - start if(sz.is_a?(TrueClass))
    read_raw(field_name, start, sz)
  end
end
