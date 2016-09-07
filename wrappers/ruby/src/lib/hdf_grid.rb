require "mtk.so"
require "narray"

# This class is used  to read an HDF-EOS grid. 
class HdfGrid

# This is the attributes for the grid, as a hash.
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
#    field_list -> Array
#
# Return a list of field name
#
  def field_list
    self.field_list_raw.split(',')
  end

# :call-seq:
#    field_rank -> Integer
#
# Return the rank of the given field
#
  def field_rank(name)
    field_size(name).length
  end
  
# Return the list of dimension of a field
  def field_dimlist(name)
    field_dimlist_raw(name).split(",")
  end

# :call-seq:
#   field_read(field_name, :start => start, :stride => stride, :edge => edge) -> NArray
#
# Read a field. This can be passed the optional arguments :start, :stride, 
# and :edge. These should be Arrays of the size field_rank(field_name) if
# passed, given the start, stride, and edge. 
# 
# :edge is the number of values to read, it is the dimension of the NArray 
# that is returned. As a convenience, if edge[i] is the 
# value true instead of an Integer, it is set to read that entire dimension
#
# If :stride is not given, the default is to use a stride of 1. If none of 
# the optional arguments are given, the default is to read the whole field. 

  def field_read(name, option = {})
    rank = field_rank(name)
    start = option[:start] || Array.new(rank, 0)
    stride = option[:stride] || Array.new(rank, 1)
    edge = option[:edge] || Array.new(rank, true)
    size = field_size(name)
    for i in 0...rank
      edge[i] = (size[i] - start[i]) / stride[i] if(edge[i].is_a?(TrueClass))
    end

# narray is in fortan order, so reverse edge because raw stuff works with C 
# order
    
    narray = case field_type(name)
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
    set_cache(name, start, stride, edge)
    field_read_raw(narray, name, start, stride, edge)

# Now, put the C array into a fortran array

    narray.transpose(*(0...narray.rank).to_a.reverse)
  end

  private

# :call-seq:
#   set_cache(field_name, start, stride, edge) -> nil
#
# Set the chunking cache to be used to read the given range of a field.
# This is set to prevent trashing of the cache. We set this so we can hold
# the whole field in cache.
#
  def set_cache (name, start, stride, edge)
    num_chunk = 1
    chunk_size = field_chunk_size(name)
    for i in 1...field_rank(name)
      num_chunk *= [(start[i] + stride[i] * edge[i]) / chunk_size[i] -
        (start[i]) / chunk_size[i], 1].max
    end
    set_chunk_cache(name, num_chunk)
  end
end
