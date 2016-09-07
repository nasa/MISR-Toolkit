# Run all tests
$:.unshift File.dirname(__FILE__)
require "test/unit"
require "mtk_region_test"
require "mtk_file_test"
require "mtk_grid_test"
require "mtk_field_test"
require "mtk_data_plane_test"
require "mtk_coordinate_test"
require "hdf_file_test"
require "hdf_vdata_test"
require "hdf_grid_test"

