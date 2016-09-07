require "mtk.so"

# This class is used to specify a Region that MtkFile then uses to
# either find orbit/paths/files that cover that region, or to read
# data from a specific file that covers the region. 
#
class MtkRegion

# Create a MtkRegion given the center and extent in degrees.
#
#  :call-seq:
#    MtkRegion.create_extent_degrees(lat, lon, lat_ext, lon_ext) -> MtkRegion
#
# Everything is in degrees
#
  def MtkRegion.create_extent_degrees(lat, lon, lat_ext, lon_ext)
    new.lat_lon_extent_degrees(lat, lon, lat_ext, lon_ext)
  end

# Create a MtkRegion given the center and extent in meters.
#
#  :call-seq:
#    MtkRegion.create_extent_meters(lat, lon, lat_ext, lon_ext) -> MtkRegion
#
# Latitude and Longitude are in degrees, extent is in meters.
#
  def MtkRegion.create_extent_meters(lat, lon, lat_ext, lon_ext)
    new.lat_lon_extent_meters(lat, lon, lat_ext, lon_ext)
  end

# Create a MtkRegion given the center and extent in pixels.
#
#  :call-seq:
#    MtkRegion.create_extent_pixel(lat, lon, resolution, lat_pixel_ext, lon_pixel_ext) -> MtkRegion
#
# Latitude and Longitude are in degrees, resolution is in meters.
#
  def MtkRegion.create_extent_pixel(lat, lon, resolution, lat_pixel_ext, lon_pixel_ext)
    new.lat_lon_extent_pixels(lat, lon, resolution, lat_pixel_ext, lon_pixel_ext)
  end

# Create a MtkRegion given the path and block range.
#
#  :call-seq:
#    MtkRegion.create_path_block_range(path, block_range) -> MtkRegion
#
# Block range is a Range.
#
  def MtkRegion.create_path_block_range(path, block_range)
    new.path_block_range(path, block_range.min, block_range.max)
  end

# Create a MtkRegion given the upper left and lower right corner.
#
#  :call-seq:
#    MtkRegion.create_corners(ulc_lat, ulc_lon, lrc_lat, lrc_lon) -> MtkRegion
#
# Latitude and Longitude are in degrees.
#
  def MtkRegion.create_corners(ulc_lat, ulc_lon, lrc_lat, lrc_lon)
    new.lat_lon_corner(ulc_lat, ulc_lon, lrc_lat, lrc_lon)
  end


  private_class_method :new
end
