require "mtk.so"
require "narray"

# This class is used  to read an HDF-EOS file. 
class HdfFile
# Open an HdfFile. If a block is passed, we execute the block passing in
# the file, and ensure that it closes when the block ends. W/o a block,
# this is just the same as Hdf.new
  def self.open(filename)
    return HdfFile.new(filename) unless block_given?
    f = nil
    begin
      f = HdfFile.new(filename)
      yield(f)
    ensure
      f.close if(f)
    end
  end

# This is the SDS attributes for the file, as a hash.
  def attributes
    unless(@attributes)
      @attributes = Hash.new
      attribute_list.each do |att|
        @attributes[att] = attribute_read(att)
      end
    end
    @attributes
  end
  
# :call-seq:
#    grid_list -> Array
#
# Return a list of grid name
#
  def grid_list
    self.grid_list_raw.split(',')
  end

# :call-seq:
#   grid(grid_name) -> HdfGrid
#
# Open a grid
# 
  def grid (name)
    HdfGrid.new(self, name)
  end

# :call-seq:
#   sds_read(field_name, :start => start, :stride => stride, :edge => edge) -> NArray
#
# Read a SDS. This can be passed the optional arguments :start, :stride, 
# and :edge. These should be Arrays of the size field_rank(field_name) if
# passed, given the start, stride, and edge. 
# 
# :edge is the number of values to read, it is the dimension of the NArray 
# that is returned. As a convenience, if edge[i] is the 
# value true instead of an Integer, it is set to read that entire dimension
#
# If :stride is not given, the default is to use a stride of 1. If none of 
# the optional arguments are given, the default is to read the whole field. 

  def sds_read(name, option = {})
    size = sds_size(name)
    rank = size.size
    start = option[:start] || Array.new(rank, 0)
    stride = option[:stride] || Array.new(rank, 1)
    edge = option[:edge] || Array.new(rank, true)
    for i in 0...rank
      edge[i] = (size[i] - start[i]) / stride[i] if(edge[i].is_a?(TrueClass))
    end

# narray is in fortan order, so reverse edge because raw stuff works with C 
# order
    
    narray = case sds_type(name)
             when 5
               NArray.sfloat(*edge.reverse)
             when 6
               NArray.float(*edge.reverse)
             when 20, 21
               NArray.byte(*edge.reverse)
             when 22,23
               NArray.sint(*edge.reverse)
             when 24, 25
               NArray.int(*edge.reverse)
             else
               throw RuntimeError("Unrecognized type")
             end
    sds_read_raw(narray, name, start, stride, edge)

# Now, put the C array into a fortran array

    narray.transpose(*(0...narray.rank).to_a.reverse)
  end

# :call-seq:
#   vdata(vdata_name) -> HdfVdata
#
# Open a vdata
#
  def vdata (name)
    HdfVdata.new(self, name)
  end

# :call-seq:
#    vdata_list -> Array
#
# Return a list of vdata names
#
  def vdata_list
    res = self.vdata_list_raw
# Strip out vdata that we don't want to list.
    return res.find_all do |i|
      !(i =~ /^_/ || 
        i =~ /HDFEOSVersion/ ||
        i =~ /^\w+:\w+$/ ||
        i =~ /^_/ ||
        i =~ /StructMetadata\.0/ ||
        i =~ /coremetadata/)
    end
  end
end
