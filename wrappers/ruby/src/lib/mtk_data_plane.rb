require "mtk.so"

# This is contains data read from a field, along with map information about
# the data
class MtkDataPlane

# :call-seq:
#   MtkDataPlane.new(data) -> MtkDataPlane
#
# Creates an object containing the given NArray of data. Note that normally
# this isn't called directly, rather MtkField#read is called to read the
# data and return a MtkDataPlane.
  def initialize(d)
    @data = d
  end

# :call-seq:
#   data -> NArray
#
# Return data that was read. This is an NArray. Data is ordered by line and 
# sample, so you can retrieve data at line L, sample S as data[L, S].
  def data
    @data
  end
end
