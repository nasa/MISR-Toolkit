/* Separate file to keep rdoc from including these in documentation */

#include "hdf_file.h"
#include "hdf_grid.h"
#include "hdf_vdata.h"
extern "C" {
#include "ruby.h"
#include "narray.h"		// Access to C part of NArray class.
#undef PACKAGE_VERSION
#undef PACKAGE_TARNAME
#undef PACKAGE_STRING	
#undef PACKAGE_NAME
#undef PACKAGE_BUGREPORT
#include "MisrToolkit.h"	// MISR toolkit
#include "stdio.h"		// for sprintf
}
#include <algorithm>		// Definition of std::copy.

/*-----------------------------------------------------------*/
/* Routines for MtkRegion.                                   */
/*-----------------------------------------------------------*/

/* call-seq:
*   region.lat_lon_extent_degrees(center_lat, center_lon, extent_lat,
extent_lon) -> region
*
* Fill in a region by defining center latitude and longitude and
* extent in latitude and longitude. All are in degrees.
*/

static VALUE mtk_region_lled(VALUE self, VALUE ctr_lat, VALUE ctr_lon,
		       VALUE ext_lat, VALUE ext_lon)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  MTKt_status status = MtkSetRegionByLatLonExtent(NUM2DBL(ctr_lat),
     NUM2DBL(ctr_lon), NUM2DBL(ext_lat), NUM2DBL(ext_lon), "degrees", r);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkSetRegionByLatLonExtent failed");
  return self;
}

/* call-seq:
*   region.lat_lon_extent_meters(center_lat, center_lon, extent_lat_meters,
extent_lon_meters) -> region
*
* Fill in a region by defining center latitude and longitude and
* extent in meters. Latitude and longitude are in degrees.
*/

static VALUE mtk_region_llem(VALUE self, VALUE ctr_lat, VALUE ctr_lon,
		       VALUE ext_lat_meters, VALUE ext_lon_meters)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  MTKt_status status = MtkSetRegionByLatLonExtent(NUM2DBL(ctr_lat),
     NUM2DBL(ctr_lon), NUM2DBL(ext_lat_meters), NUM2DBL(ext_lon_meters),
     "meters", r);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkSetRegionByLatLonExtent failed");
  return self;
}

/* call-seq:
*   region.lat_lon_extent_pixel(center_lat, center_lon, resolution, extent_pixel_lat,
extent_pixel_lon) -> region
*
* Fill in a region by defining center latitude and longitude and
* extent in pixels. Latitude and longitude are in degrees, resolution
* is in meters
*/

static VALUE mtk_region_llep(VALUE self, VALUE ctr_lat, VALUE ctr_lon,
			     VALUE resolution, VALUE ext_lat, VALUE ext_lon)
{
  MTKt_Region* r;
  char units[20];
  Data_Get_Struct(self, MTKt_Region, r);
  sprintf(units, "%ldm",  NUM2INT(resolution));
  MTKt_status status = MtkSetRegionByLatLonExtent(NUM2DBL(ctr_lat),
	NUM2DBL(ctr_lon), NUM2INT(ext_lat), NUM2INT(ext_lon), units, r);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkSetRegionByLatLonExtent failed");
  return self;
}

/* call-seq:
*   region.path_block_range(path, start_block, end_block) -> region
*
* Fill in a region by giving path, start and end block (start and end
* are inclusive).
*/

static VALUE mtk_region_pbr(VALUE self, VALUE path, VALUE start, VALUE end)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  MTKt_status status = MtkSetRegionByPathBlockRange(NUM2INT(path),
     NUM2INT(start), NUM2INT(end), r);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkSetRegionByPathBlockRange failed");
  return self;
}

/* call-seq:
*   region.lat_lon_extent_degrees(ulc_lat, ulc_lon, lrc_lat, lrc_lon) -> region
*
* Fill in a region by giving upper left and lower right corner. All
* are in degrees. 
*/

static VALUE mtk_region_llcn(VALUE self, VALUE ulc_lat, VALUE ulc_lon,
		       VALUE lrc_lat, VALUE lrc_lon)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  MTKt_status status = MtkSetRegionByUlcLrc(NUM2DBL(ulc_lat),
     NUM2DBL(ulc_lon), NUM2DBL(lrc_lat), NUM2DBL(lrc_lon), r);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkSetRegionByUlcLrc failed");
  return self;
}

/*-----------------------------------------------------------*/
/* Routines for MtkFile.                                     */
/*-----------------------------------------------------------*/

/* call-seq:
 *   MtkFile.file_to_path(Filename) -> int
 *
 * Return path for given file.
 */

static VALUE mtk_file_file_to_path(VALUE klass, VALUE fname)
{
  int path;
  MTKt_status status = MtkFileToPath(StringValuePtr(fname), &path);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileToPath failed");
  return INT2NUM(path);
}

/* call-seq:
 *   MtkFile.file_to_local_granule_id(Filename) -> String
 *
 * Return local granule ID for given file.
 */

static VALUE mtk_file_file_to_local_granule_id(VALUE klass, VALUE fname)
{
  char* lgid;
  MTKt_status status = MtkFileLGID(StringValuePtr(fname), &lgid);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileLGID failed");
  VALUE res = rb_str_new2(lgid);
  free(lgid);
  return res;
}

/* call-seq:
 *   MtkFile.file_to_file_type(Filename) -> Integer
 *
 * Return file type.
 */

static VALUE mtk_file_file_to_file_type(VALUE klass, VALUE fname)
{
  MTKt_FileType ft;
  MTKt_status status = MtkFileType(StringValuePtr(fname), &ft);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileType failed");
  return INT2NUM(ft);
}

/* call-seq:
 *   MtkFile.file_to_version(Filename) -> String
 *
 * Return file version.
 */

static VALUE mtk_file_file_to_version(VALUE klass, VALUE fname)
{
  char fv[100];
  MTKt_status status = MtkFileVersion(StringValuePtr(fname), fv);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileVersion failed");
  return rb_str_new2(fv);;
}

/* call-seq:
 *   MtkFile.file_to_block(Filename) -> Range
 *
 * Return block range for given file.
 */

static VALUE mtk_file_file_to_block(VALUE klass, VALUE fname)
{
  int start_block, end_block;
  MTKt_status status = MtkFileToBlockRange(StringValuePtr(fname), &start_block, 
				     &end_block);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileToBlockRange failed");
  VALUE rarg[2];
  rarg[0] = INT2NUM(start_block);
  rarg[1] = INT2NUM(end_block);
  VALUE res = rb_class_new_instance(2, rarg, 
	    rb_const_get(rb_cObject, rb_intern("Range")));
  return res;
}

/* call-seq:
 *   MtkFile.file_to_grid_list(Filename) -> Array
 *
 * Return Array of grid name Strings of given file.
 */

static VALUE mtk_file_file_to_grid_list(VALUE klass, VALUE fname)
{
  char** grid_list;
  int grid_length;
  MTKt_status status = MtkFileToGridList(StringValuePtr(fname), &grid_length,
					 &grid_list);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileToGridList failed");
  VALUE res = rb_ary_new();
  for(int i = 0; i < grid_length; ++i) {
    rb_ary_push(res, rb_str_new2(grid_list[i]));
    free(grid_list[i]);
  }
  free(grid_list);
  return res;
}

/*-----------------------------------------------------------*/
/* Routines for MtkGrid.                                     */
/*-----------------------------------------------------------*/

/* call-seq:
 *   MtkGrid.grid_to_field_list(Filename, Grid_name) -> Array
 *
 * Return Array of field list name Strings of a given file
 */
static VALUE mtk_grid_grid_to_field_list(VALUE klass, VALUE fname, VALUE grid)
{
  char** field_list;
  int field_length;
  MTKt_status status = MtkFileGridToFieldList(StringValuePtr(fname), 
					      StringValuePtr(grid), 
					      &field_length, &field_list);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileGridToFieldList failed");
  VALUE res = rb_ary_new();
  for(int i = 0; i < field_length; ++i) {
    rb_ary_push(res, rb_str_new2(field_list[i]));
    free(field_list[i]);
  }
  free(field_list);
  return res;
}

/* call-seq:
 *   MtkGrid.grid_to_resolution(Filename, Grid_name) -> Integer
 *
 * Return resolution of grid.
 */

static VALUE mtk_grid_grid_to_resolution(VALUE klass, VALUE fname, VALUE grid)
{
  int resolution;
  MTKt_status status = MtkFileGridToResolution(StringValuePtr(fname), 
					       StringValuePtr(grid), 
					       &resolution);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileGridToResolution failed");
  return INT2NUM(resolution);
}

/*-----------------------------------------------------------*/
/* Routines for MtkField.                                   */
/*-----------------------------------------------------------*/

/* call-seq:
 *   MtkField.field_to_datatype(Filename, Grid_name, Field_name) -> Integer
 *
 * Return datatype of field
 */

static VALUE mtk_field_field_to_datatype(VALUE klass, VALUE fname, 
					 VALUE grid, VALUE field)
{
  MTKt_DataType data_type;
  MTKt_status status = 
    MtkFileGridFieldToDataType(StringValuePtr(fname), 
			       StringValuePtr(grid), 
			       StringValuePtr(field),
			       &data_type);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileGridFieldToDataType failed");
  return INT2NUM((int) data_type);
}

/* call-seq:
 *   MtkGrid.field_to_dimlist(Filename, Grid_name, Field_name) -> [[Dim name list], [Size dim]]
 *
 * Return list of dimensions and size of each dimension.
 */

static VALUE mtk_grid_field_to_dimlist(VALUE klass, VALUE fname, 
				       VALUE grid, VALUE field)
{
  int dimcnt;
  char **dimlist;
  int *dimsize;
  MTKt_status status = 
    MtkFileGridFieldToDimList(StringValuePtr(fname), 
			      StringValuePtr(grid), 
			      StringValuePtr(field),
			      &dimcnt, &dimlist, &dimsize);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFileGridFieldToDimList failed");
  VALUE res1 = rb_ary_new();
  VALUE res2 = rb_ary_new();
  for(int i = 0; i < dimcnt; ++i) {
    rb_ary_push(res1, rb_str_new2(dimlist[i]));
    rb_ary_push(res2, INT2NUM(dimsize[i]));
  }
  VALUE res = rb_ary_new();
  rb_ary_push(res, res1);
  rb_ary_push(res, res2);
  MtkStringListFree(dimcnt, &dimlist);
  free(dimsize);
  return res;
}

/* call-seq:
 *   MtkField.field_to_fillvalue(Filename, Grid_name, Field_name) -> Number
 *
 * Return fill type of field
 */

static VALUE mtk_field_field_to_fillvalue(VALUE klass, VALUE fname, 
					  VALUE grid, VALUE field)
{
  MTKt_DataBuffer fillbuf;
  MTKt_status status = 
    MtkFillValueGet(StringValuePtr(fname), 
		    StringValuePtr(grid), 
		    StringValuePtr(field),
		    &fillbuf);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkFillValueGet failed");
  VALUE res;
  switch(fillbuf.datatype) {
  case MTKe_char8: 
    res = LONG2NUM(fillbuf.data.c8[0][0]);
    break;
  case MTKe_uchar8:
    res = LONG2NUM(fillbuf.data.uc8[0][0]);
    break;
  case MTKe_int8:	
    res = LONG2NUM(fillbuf.data.i8[0][0]);
    break;
  case MTKe_uint8:
    res = LONG2NUM(fillbuf.data.u8[0][0]);
    break;
  case MTKe_int16:
    res = LONG2NUM(fillbuf.data.i16[0][0]);
    break;
  case MTKe_uint16:
    res = LONG2NUM(fillbuf.data.u16[0][0]);
    break;
  case MTKe_int32: 	
    res = LONG2NUM(fillbuf.data.i32[0][0]);
    break;
  case MTKe_uint32:
    res = LONG2NUM(fillbuf.data.u32[0][0]);
    break;
  case MTKe_int64: 	
    res = LONG2NUM(fillbuf.data.i64[0][0]);
    break;
  case MTKe_uint64:
    res = LONG2NUM(fillbuf.data.u64[0][0]);
    break;
  case MTKe_float: 	
    res = rb_float_new(fillbuf.data.f[0][0]);
    break;
  case MTKe_double:
    res = rb_float_new(fillbuf.data.d[0][0]);
    break;
  default:
    rb_raise(rb_eRuntimeError, "Unrecognized data type");
  }
  MtkDataBufferFree(&fillbuf);
  return res;
}

// Helper routine

template<class P, class T> VALUE create_narray(NArray_Types Type, 
					       P** Data, 
					       int NLine, int NSamp, 
					       T Dummy)
{
  VALUE narg[3];
  narg[0] = INT2NUM(Type);
  narg[1] = INT2NUM(NLine);
  narg[2] = INT2NUM(NSamp);
  VALUE res = rb_funcall2(rb_const_get(rb_cObject, rb_intern("NArray")),
			  rb_intern("new"), 3, narg);
  struct NARRAY* narray;
  GetNArray(res, narray);
  T* ndata = reinterpret_cast<T*>(NA_PTR(narray, 0));

// We need to be carful here. NArray is is column major form (like
// fortran), while the data from the MISR toolkit is normal C row
// major form.

  for(int c = 0; c < NSamp; ++c)
    for(int r = 0; r < NLine; ++r)
      *ndata++ = static_cast<P>(Data[r][c]);
  return res;
}

/* call-seq:
 *   MtkField.field_read(Filename, Grid_name, Field_name, Region) -> MtkDataPlane
 *
 * Read given region.
 */

static VALUE mtk_field_read(VALUE klass, VALUE fname, 
			    VALUE grid, VALUE field, VALUE region)
{
  MTKt_Region* r;
  Data_Get_Struct(region, MTKt_Region, r);
  MTKt_DataBuffer fillbuf;
  MTKt_MapInfo map_info = MTKT_MAPINFO_INIT;
  MTKt_status status = MtkReadData(StringValuePtr(fname), 
				   StringValuePtr(grid), 
				   StringValuePtr(field),
				   *r,
				   &fillbuf,
				   &map_info);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkReadData failed");
  VALUE data;
  switch(fillbuf.datatype) {
  case MTKe_char8:
    data = create_narray(NA_BYTE, fillbuf.data.c8, fillbuf.nline, 
			fillbuf.nsample, (u_int8_t) 0);
    break;
  case MTKe_uchar8:
    data = create_narray(NA_BYTE, fillbuf.data.uc8, fillbuf.nline, 
			fillbuf.nsample, (u_int8_t) 0); 
    break;
  case MTKe_int8:	
    data = create_narray(NA_SINT, fillbuf.data.i8, fillbuf.nline, 
			fillbuf.nsample, (int16_t) 0);
    break;
  case MTKe_uint8:
    data = create_narray(NA_BYTE, fillbuf.data.u8, fillbuf.nline, 
			fillbuf.nsample, (u_int8_t) 0); 
    break;
  case MTKe_int16:
    data = create_narray(NA_SINT, fillbuf.data.i16, fillbuf.nline, 
			fillbuf.nsample, (int16_t) 0);
    break;
  case MTKe_uint16:
    data = create_narray(NA_LINT, fillbuf.data.u16, fillbuf.nline, 
			fillbuf.nsample, (u_int32_t) 0);
    break;
  case MTKe_int32: 	
    data = create_narray(NA_LINT, fillbuf.data.i32, fillbuf.nline, 
			fillbuf.nsample, (int32_t) 0);
    break;
  case MTKe_uint32:
    data = create_narray(NA_LINT, fillbuf.data.u32, fillbuf.nline, 
			fillbuf.nsample, (u_int32_t) 0);
    break;
  case MTKe_int64: 	
    rb_raise(rb_eRuntimeError, 
	     "Type INT64 isn't supported, because NArray doesn't support it");
    break;
  case MTKe_uint64:
    rb_raise(rb_eRuntimeError, 
	     "Type UINT64 isn't supported, because NArray doesn't support it");
    break;
  case MTKe_float: 	
    data = create_narray(NA_SFLOAT, fillbuf.data.f, fillbuf.nline, 
			fillbuf.nsample, (float) 0);
    break;
  case MTKe_double:
    data = create_narray(NA_DFLOAT, fillbuf.data.d, fillbuf.nline, 
			fillbuf.nsample, (double) 0);
    break;
  default:
    rb_raise(rb_eRuntimeError, "Unrecognized data type");
  }
  MtkDataBufferFree(&fillbuf);
  VALUE res = rb_funcall(rb_const_get(rb_cObject, rb_intern("MtkDataPlane")),
			 rb_intern("new"), 1, data);
  MTKt_MapInfo* mi;
  Data_Get_Struct(res, MTKt_MapInfo, mi);
  *mi = map_info;
  return res;
}

/*-----------------------------------------------------------*/
/* Routines for HdfFile.                                     */
/*-----------------------------------------------------------*/

/* call-seq: 
 *  grid_list_raw ->  string
 *
 * Return a comma seperated list of grid names.
 *
 */
static VALUE hdf_grid_list_raw(VALUE self)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    std::string s = h->grid_list();
    return rb_str_new2(s.c_str());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  vdata_list_raw ->  string
 *
 * Return a comma seperated list of vdata names.
 *
 */
static VALUE hdf_vdata_list_raw(VALUE self)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    std::set<std::string> res = h->vdata_list();
    VALUE res_val = rb_ary_new();
    for(std::set<std::string>::const_iterator i = res.begin(); 
	i != res.end(); ++i)
      rb_ary_push(res_val, rb_str_new2((*i).c_str()));
    return res_val;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  sds_type(field_name) ->  Integer
 *
 * Return the field type
 *
 */
static VALUE sds_type(VALUE self, VALUE name_obj)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    return INT2FIX(h->sds_data_type(RSTRING(StringValue(name_obj))->ptr));
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  sds_read_raw(out, field_name, start, strid, edge) -> nil
 *
 * Read field into the already allocated NArray out.
 *
 */
static VALUE sds_read_raw(VALUE self, VALUE narray_obj, VALUE name_obj,
		          VALUE start_obj, VALUE stride_obj,
			  VALUE edge_obj)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    const std::vector<int32>& sds_dim = 
      h->sds_dim(RSTRING(StringValue(name_obj))->ptr);
    struct NARRAY* narray;
    GetNArray(narray_obj, narray);
    std::vector<int32> start(RARRAY(start_obj)->len),
      stride(RARRAY(stride_obj)->len), edge(RARRAY(edge_obj)->len);
    if(start.size() != stride.size() ||
       start.size() != edge.size() ||
       start.size() != sds_dim.size())
      throw Exception("Start, stride, and edge don't all match the rank of the field");
    for(int i = 0; i < (int) start.size(); ++i) {
      start[i] = NUM2INT(RARRAY(start_obj)->ptr[i]);
      stride[i] = NUM2INT(RARRAY(stride_obj)->ptr[i]);
      edge[i] = NUM2INT(RARRAY(edge_obj)->ptr[i]);
    }
    h->sds_read(RSTRING(StringValue(name_obj))->ptr, narray->ptr, &(start[0]),
		  &(stride[0]), &(edge[0]));
    return Qnil;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   attribute_list -> Array of Strings
 *
 * Return the list of SDS attributes in the file.
 *
 */

static VALUE attribute_list(VALUE self)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    VALUE res = rb_ary_new();
    for(int i = 0; i < (int) h->attribute_list().size(); ++i)
      rb_funcall(res, rb_intern("push"), 1, rb_str_new2(h->attribute_list()[i].c_str()));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   attribute_read(attribute) -> Value
 *
 * Return the value of the given attribute.
 *
 */

template<class T> VALUE to_val_long(const std::vector<T>& V)
{
  VALUE res = rb_ary_new();
  for(int i = 0; i < (int) V.size(); ++i)
    rb_funcall(res, rb_intern("push"), 1, LONG2NUM(V[i]));
  return res;
}

template<class T> VALUE to_val_float(const std::vector<T>& V)
{
  VALUE res = rb_ary_new();
  for(int i = 0; i < (int) V.size(); ++i)
    rb_funcall(res, rb_intern("push"), 1, rb_float_new(V[i]));
  return res;
}

static VALUE attribute_read(VALUE self, VALUE att_name)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    std::string astr = RSTRING(StringValue(att_name))->ptr;
    int32 type = h->attribute_data_type(astr);
    int32 size = h->attribute_data_size(astr);
    VALUE res;
    if(size ==1)
      switch(type) {
      case DFNT_FLOAT32:
	res = rb_float_new(h->read_attribute<float32>(astr));
	break;
      case DFNT_FLOAT64:
	res = rb_float_new(h->read_attribute<float64>(astr));
	break;
      case DFNT_INT8:
	res = LONG2NUM(h->read_attribute<int8>(astr));
	break;
      case DFNT_UINT8:
	res = LONG2NUM(h->read_attribute<uint8>(astr));
	break;
      case DFNT_INT16:
	res = LONG2NUM(h->read_attribute<int16>(astr));
	break;
      case DFNT_UINT16:
	res = LONG2NUM(h->read_attribute<uint16>(astr));
	break;
      case DFNT_INT32:
	res = LONG2NUM(h->read_attribute<int32>(astr));
	break;
      case DFNT_UINT32:
	res = LONG2NUM(h->read_attribute<uint32>(astr));
	break;
      case DFNT_CHAR8:
	res = LONG2NUM(h->read_attribute<int8>(astr));
	break;
      case DFNT_UCHAR8:
	res = LONG2NUM(h->read_attribute<uint8>(astr));
	break;
      default:
	rb_raise(rb_eRuntimeError, "Unrecognized data type");
      }
    else
      switch(type) {
      case DFNT_FLOAT32:
	res = to_val_float(h->read_attribute<std::vector<float32> >(astr));
	break;
      case DFNT_FLOAT64:
	res = to_val_float(h->read_attribute<std::vector<float64> >(astr));
	break;
      case DFNT_INT8:
	res = to_val_long(h->read_attribute<std::vector<int8> >(astr));
	break;
      case DFNT_UINT8:
	res = to_val_long(h->read_attribute<std::vector<uint8> >(astr));
	break;
      case DFNT_INT16:
	res = to_val_long(h->read_attribute<std::vector<int16> >(astr));
	break;
      case DFNT_UINT16:
	res = to_val_long(h->read_attribute<std::vector<uint16> >(astr));
	break;
      case DFNT_INT32:
	res = to_val_long(h->read_attribute<std::vector<int32> >(astr));
	break;
      case DFNT_UINT32:
	res = to_val_long(h->read_attribute<std::vector<uint32> >(astr));
	break;
      case DFNT_CHAR8:
	res = rb_str_new2(h->read_attribute<std::string>(astr).c_str());
	break;
      case DFNT_UCHAR8:
	res = to_val_long(h->read_attribute<std::vector<uint8> >(astr));
	break;
      default:
	rb_raise(rb_eRuntimeError, "Unrecognized data type");
      }
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/*-----------------------------------------------------------*/
/* Routines for HdfGrid.                                     */
/*-----------------------------------------------------------*/

/* call-seq: 
 *  field_list_raw ->  string
 *
 * Return a comma seperated list of field names.
 *
 */
static VALUE grid_field_list_raw(VALUE self)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    std::string s = g->field_list();
    return rb_str_new2(s.c_str());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  field_chunk_size(field_name) ->  Array
 *
 * Return the chunking information about a field
 *
 */
static VALUE grid_field_chunk_size(VALUE self, VALUE name_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->populate_field_info(RSTRING(StringValue(name_obj))->ptr);
    VALUE res = rb_ary_new();
    for(int i = 0; i < g->rank(); ++i)
      if(g->tile_code() ==HDFE_NOTILE)
	rb_funcall(res, rb_intern("push"), 1, INT2FIX(g->dim()[i]));
      else
	rb_funcall(res, rb_intern("push"), 1, INT2FIX(g->tile_dim()[i]));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  field_type(field_name) ->  Integer
 *
 * Return the field type
 *
 */
static VALUE grid_field_type(VALUE self, VALUE name_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->populate_field_info(RSTRING(StringValue(name_obj))->ptr);
    return INT2FIX(g->data_type());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  field_dimlist_raw(field_name) ->  String
 *
 * Return the dimension list for the field.
 *
 */
static VALUE grid_field_dimlist_raw(VALUE self, VALUE name_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->populate_field_info(RSTRING(StringValue(name_obj))->ptr);
    return rb_str_new2(g->dim_name_buf());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  set_chunk_cache(field_name, Number chunks) ->  Nil
 *
 * Set the cache used for the given field
 *
 */
static VALUE grid_set_chunk_cache(VALUE self, VALUE name_obj, 
				  VALUE Number_chunk)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->set_chunk_cache(RSTRING(StringValue(name_obj))->ptr,
		       NUM2INT(Number_chunk));
    return Qnil;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq: 
 *  grid_field_read_raw(out, field_name, start, strid, edge) -> nil
 *
 * Read field into the already allocated NArray out.
 *
 */
static VALUE grid_field_read_raw(VALUE self, VALUE narray_obj, VALUE name_obj,
				 VALUE start_obj, VALUE stride_obj,
				 VALUE edge_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->populate_field_info(RSTRING(StringValue(name_obj))->ptr);
    struct NARRAY* narray;
    GetNArray(narray_obj, narray);
    std::vector<int32> start(RARRAY(start_obj)->len),
      stride(RARRAY(stride_obj)->len), edge(RARRAY(edge_obj)->len);
    if(start.size() != stride.size() ||
       start.size() != edge.size() ||
       (int32) start.size() != g->rank())
      throw Exception("Start, stride, and edge don't all match the rank of the field");
    for(int i = 0; i < (int) start.size(); ++i) {
      start[i] = NUM2INT(RARRAY(start_obj)->ptr[i]);
      stride[i] = NUM2INT(RARRAY(stride_obj)->ptr[i]);
      edge[i] = NUM2INT(RARRAY(edge_obj)->ptr[i]);
    }
    g->field_read(RSTRING(StringValue(name_obj))->ptr, narray->ptr, &(start[0]),
		  &(stride[0]), &(edge[0]));
    return Qnil;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   grid.attribute_list -> Array of Strings
 *
 * Return the list of attributes in the grid.
 *
 */

static VALUE grid_attribute_list(VALUE self)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    VALUE res = rb_ary_new();
    for(int i = 0; i < (int) g->attribute_list().size(); ++i)
      rb_funcall(res, rb_intern("push"), 1, rb_str_new2(g->attribute_list()[i].c_str()));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

static VALUE grid_attribute_read(VALUE self, VALUE att_name)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    std::string astr = RSTRING(StringValue(att_name))->ptr;
    int32 type = g->attribute_data_type(astr);
    int32 size = g->attribute_data_size(astr);
    VALUE res;
    if(size ==1)
      switch(type) {
      case DFNT_FLOAT32:
	res = rb_float_new(g->read_attribute<float32>(astr));
	break;
      case DFNT_FLOAT64:
	res = rb_float_new(g->read_attribute<float64>(astr));
	break;
      case DFNT_INT8:
	res = LONG2NUM(g->read_attribute<int8>(astr));
	break;
      case DFNT_UINT8:
	res = LONG2NUM(g->read_attribute<uint8>(astr));
	break;
      case DFNT_INT16:
	res = LONG2NUM(g->read_attribute<int16>(astr));
	break;
      case DFNT_UINT16:
	res = LONG2NUM(g->read_attribute<uint16>(astr));
	break;
      case DFNT_INT32:
	res = LONG2NUM(g->read_attribute<int32>(astr));
	break;
      case DFNT_UINT32:
	res = LONG2NUM(g->read_attribute<uint32>(astr));
	break;
      case DFNT_CHAR8:
	res = LONG2NUM(g->read_attribute<int8>(astr));
	break;
      case DFNT_UCHAR8:
	res = LONG2NUM(g->read_attribute<uint8>(astr));
	break;
      default:
	rb_raise(rb_eRuntimeError, "Unrecognized data type");
      }
    else
      switch(type) {
      case DFNT_FLOAT32:
	res = to_val_float(g->read_attribute<std::vector<float32> >(astr));
	break;
      case DFNT_FLOAT64:
	res = to_val_float(g->read_attribute<std::vector<float64> >(astr));
	break;
      case DFNT_INT8:
	res = to_val_long(g->read_attribute<std::vector<int8> >(astr));
	break;
      case DFNT_UINT8:
	res = to_val_long(g->read_attribute<std::vector<uint8> >(astr));
	break;
      case DFNT_INT16:
	res = to_val_long(g->read_attribute<std::vector<int16> >(astr));
	break;
      case DFNT_UINT16:
	res = to_val_long(g->read_attribute<std::vector<uint16> >(astr));
	break;
      case DFNT_INT32:
	res = to_val_long(g->read_attribute<std::vector<int32> >(astr));
	break;
      case DFNT_UINT32:
	res = to_val_long(g->read_attribute<std::vector<uint32> >(astr));
	break;
      case DFNT_CHAR8:
	res = rb_str_new2(g->read_attribute<std::string>(astr).c_str());
	break;
      case DFNT_UCHAR8:
	res = to_val_long(g->read_attribute<std::vector<uint8> >(astr));
	break;
      default:
	rb_raise(rb_eRuntimeError, "Unrecognized data type");
      }
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}


/*-----------------------------------------------------------*/
/* Routines for HdfVdata.                                     */
/*-----------------------------------------------------------*/

/* call-seq: 
 *  field_list_raw ->  string
 *
 * Return a comma seperated list of field names.
 *
 */
static VALUE vdata_field_list_raw(VALUE self)
{
  try {
    HdfVdata* v;
    Data_Get_Struct(self, HdfVdata, v);
    std::string s = v->field_list();
    return rb_str_new2(s.c_str());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   read_raw("Field name", Start_record, Number_record) -> Array of values
 *
 * Read the given vdata field, from start record for the given number
 * of records
 *
 */
static VALUE vdata_read(VALUE self, VALUE field_name, VALUE rec_i, 
			VALUE num_read)
{
  try {
    HdfVdata* v;
    Data_Get_Struct(self, HdfVdata, v);
    return v->read(RSTRING(StringValue(field_name))->ptr, 
		   NUM2INT(rec_i), NUM2INT(num_read));
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}



/*-----------------------------------------------------------*/
/* Initialize class                                          */
/*-----------------------------------------------------------*/

void init_mtk_private(VALUE mtk_region_class, VALUE mtk_file_class, 
		      VALUE mtk_grid_class, VALUE mtk_field_class,
		      VALUE mtk_data_plane_class, VALUE hdf_class,
		      VALUE grid_class, VALUE vdata_class) 
{
  rb_define_method(hdf_class, "grid_list_raw", (VALUE (*)(...)) hdf_grid_list_raw, 0);
  rb_define_method(hdf_class, "vdata_list_raw", (VALUE (*)(...)) hdf_vdata_list_raw, 0);
  rb_define_method(hdf_class, "sds_type", (VALUE (*)(...)) sds_type, 1);
  rb_define_method(hdf_class, "attribute_list", 
		   (VALUE (*)(...)) attribute_list, 0);
  rb_define_method(hdf_class, "attribute_read", 
		   (VALUE (*)(...)) attribute_read, 1);
  rb_define_method(hdf_class, "sds_read_raw", (VALUE (*)(...)) sds_read_raw, 5);
  rb_define_method(grid_class, "attribute_list", (VALUE (*)(...)) grid_attribute_list, 0);
  rb_define_method(grid_class, "attribute_read", 
		   (VALUE (*)(...)) grid_attribute_read, 1);
  rb_define_method(grid_class, "field_list_raw", (VALUE (*)(...)) grid_field_list_raw, 0);
  rb_define_method(grid_class, "field_chunk_size", (VALUE (*)(...)) grid_field_chunk_size, 1);
  rb_define_method(grid_class, "set_chunk_cache", 
	    (VALUE (*)(...)) grid_set_chunk_cache, 2);
  rb_define_method(grid_class, "field_type", (VALUE (*)(...)) grid_field_type, 1);
  rb_define_method(grid_class, "field_dimlist_raw", (VALUE (*)(...)) grid_field_dimlist_raw, 1);
  rb_define_method(grid_class, "field_read_raw", (VALUE (*)(...)) grid_field_read_raw, 5);
  rb_define_method(vdata_class, "field_list_raw", (VALUE (*)(...)) vdata_field_list_raw,
		   0);
  rb_define_method(vdata_class, "read_raw", (VALUE (*)(...)) vdata_read, 3);
  rb_define_method(mtk_region_class, "lat_lon_extent_degrees", 
		   (VALUE (*)(...)) mtk_region_lled, 4);
  rb_define_method(mtk_region_class, "lat_lon_extent_meters", 
		   (VALUE (*)(...)) mtk_region_llem, 4);
  rb_define_method(mtk_region_class, "lat_lon_extent_pixels", 
		  (VALUE (*)(...)) mtk_region_llep, 5);
  rb_define_method(mtk_region_class, "path_block_range", 
		   (VALUE (*)(...)) mtk_region_pbr, 3);
  rb_define_method(mtk_region_class, "lat_lon_corner", 
		   (VALUE (*)(...)) mtk_region_llcn, 4);
  rb_define_singleton_method(mtk_file_class, "file_to_path", 
			     (VALUE (*)(...)) mtk_file_file_to_path, 1);
  rb_define_singleton_method(mtk_file_class, "file_to_block", 
			     (VALUE (*)(...)) mtk_file_file_to_block, 1);
  rb_define_singleton_method(mtk_file_class, "file_to_grid_list", 
			     (VALUE (*)(...)) mtk_file_file_to_grid_list, 1);
  rb_define_singleton_method(mtk_file_class, "file_to_local_granule_id", 
			     (VALUE (*)(...)) mtk_file_file_to_local_granule_id, 1);
  rb_define_singleton_method(mtk_file_class, "file_to_file_type", 
			     (VALUE (*)(...)) mtk_file_file_to_file_type, 1);
  rb_define_singleton_method(mtk_file_class, "file_to_version", 
			     (VALUE (*)(...)) mtk_file_file_to_version, 1);
  rb_define_singleton_method(mtk_grid_class, "grid_to_field_list", 
			     (VALUE (*)(...)) mtk_grid_grid_to_field_list, 2);
  rb_define_singleton_method(mtk_grid_class, "grid_to_resolution", 
			     (VALUE (*)(...)) mtk_grid_grid_to_resolution, 2);
  rb_define_singleton_method(mtk_grid_class, "field_to_dimlist", 
			     (VALUE (*)(...)) mtk_grid_field_to_dimlist, 3);
  rb_define_singleton_method(mtk_field_class, "field_to_datatype", 
			     (VALUE (*)(...)) mtk_field_field_to_datatype, 3);
  rb_define_singleton_method(mtk_field_class, "field_to_fillvalue", 
			     (VALUE (*)(...)) mtk_field_field_to_fillvalue, 3);
  rb_define_singleton_method(mtk_field_class, "field_read", 
			     (VALUE (*)(...)) mtk_field_read, 4);
}
