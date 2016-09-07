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
void Init_mtk();
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

/*-----------------------------------------------------------*/
/* Allocation routines for various classes.                  */
/*-----------------------------------------------------------*/

/* Free routine for MTKt_Region */
static void mtk_region_free(void *r) {
  delete (MTKt_Region *) r;
}

/* Allocator for MTKt_Region */
static VALUE mtk_region_alloc(VALUE klass) {
  MTKt_Region* r = new MTKt_Region;
  if(r)
    return Data_Wrap_Struct(klass, 0, mtk_region_free, r);
  else
    rb_raise(rb_eRuntimeError, "Trouble allocating space for MTKt_Region object");
}

static VALUE mtk_region_copy(VALUE copy, VALUE orig) {
  rb_raise(rb_eRuntimeError, "Copy/Clone not supported yet");
}

/* Free routine for MtkDataPlane */
static void mtk_data_plane_free(void *r) {
  delete (MTKt_MapInfo *) r;
}

/* Allocator for MTKtDataPlane */
static VALUE mtk_data_plane_alloc(VALUE klass) {
  MTKt_MapInfo* r = new MTKt_MapInfo;
  if(r)
    return Data_Wrap_Struct(klass, 0, mtk_data_plane_free, r);
  else
    rb_raise(rb_eRuntimeError, "Trouble allocating space for MTKt_MapInfo object");
}

static VALUE mtk_data_plane_copy(VALUE copy, VALUE orig) {
  rb_raise(rb_eRuntimeError, "Copy/Clone not supported yet");
}

/* Free routine for HdfFile*/
static void hdf_free(void *p) {
#ifdef DEBUG
  std::cerr << "Destroying HdfFile " << (HdfFile*) p << "\n";
#endif
  delete (HdfFile *) p;
}

/* Allocation for HdfFild */
static VALUE hdf_alloc(VALUE klass) {
  try {
    HdfFile* p = new HdfFile();
#ifdef DEBUG
  std::cerr << "Creating HdfFile " << (HdfFile*) p << "\n";
#endif
    VALUE res = Data_Wrap_Struct(klass, 0, hdf_free, p);
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

static VALUE hdf_copy(VALUE copy, VALUE orig) {
  rb_raise(rb_eRuntimeError, "Copy/Clone not supported yet");
}

/* Free for HdfGrid */
static void grid_free(void *p) {
#ifdef DEBUG
  std::cerr << "Destroying HdfGrid " << (HdfGrid*) p << "\n";
#endif
  delete (HdfGrid *) p;
}

/* Mark for garbage collector for HdfGrid */
static void grid_mark(void *p) {
  rb_gc_mark(((HdfGrid *) p)->hdf_obj());
}

/* Allocator for HdfGrid */
static VALUE grid_alloc(VALUE klass) {
  try {
    HdfGrid* p = new HdfGrid();
#ifdef DEBUG
  std::cerr << "Creating HdfGrid " << (HdfGrid*) p << "\n";
#endif
    return Data_Wrap_Struct(klass, grid_mark, grid_free, p);
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

static VALUE grid_copy(VALUE copy, VALUE orig) {
  rb_raise(rb_eRuntimeError, "Copy/Clone not supported yet");
}


/* Free for HdfVdata */
static void vdata_free(void *p) {
#ifdef DEBUG
  std::cerr << "Destroying HdfVdata " << (HdfVdata*) p << "\n";
#endif
  delete (HdfVdata *) p;
}

/* Marker for garbage collector for HdfVdata */
static void vdata_mark(void *p) {
  rb_gc_mark(((HdfVdata *) p)->hdf_obj());
}


/* Allocator for Vdata */
static VALUE vdata_alloc(VALUE klass) {
  try {
    HdfVdata* p = new HdfVdata();
#ifdef DEBUG
  std::cerr << "Creating HdfVdata " << (HdfVdata*) p << "\n";
#endif
    return Data_Wrap_Struct(klass, vdata_mark, vdata_free, p);
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

static VALUE vdata_copy(VALUE copy, VALUE orig) {
  rb_raise(rb_eRuntimeError, "Copy/Clone not supported yet");
}

/*-----------------------------------------------------------*/
/* Routines for MtkDataPlane                                 */
/*-----------------------------------------------------------*/

/* call-seq:
 *  lat_lon_to_line_sample(lat_lon) -> NArray
 *
 * This takes a N x 2 Narray of latitude and longitude points (as
 * decimal degrees, latitude first and longitude second). It returns a
 * N x 2 Narray of Line/Sample points that goes with that latitude
 * longitude point.
 *
 * For points that are outside of the line/sample range of the
 * MtkDataPlane, we return a line/sample of -1.
 */

static VALUE mtk_data_plane_lat_lon_to_line_sample(VALUE self, VALUE lat_lon_v)
{
  MTKt_MapInfo* r = new MTKt_MapInfo;
  Data_Get_Struct(self, MTKt_MapInfo, r);
  struct NARRAY* lat_lon;
  GetNArray(lat_lon_v, lat_lon);
  if(lat_lon->rank != 2)
    rb_raise(rb_eRuntimeError, "Lat/Lon array must be rank 2");
  if(lat_lon->shape[1] != 2)
    rb_raise(rb_eRuntimeError, "Lat/Lon array must have second dimension of 2");
  if(lat_lon->type != NA_DFLOAT)
    rb_raise(rb_eRuntimeError, "Lat/Lon array needs to be type DFLOAT");
  int n = lat_lon->shape[0];
  VALUE narg[3];
  narg[0] = INT2NUM(NA_SFLOAT);
  narg[1] = INT2NUM(n);
  narg[2] = INT2NUM(2);
  VALUE res = rb_funcall2(rb_const_get(rb_cObject, rb_intern("NArray")),
			  rb_intern("new"), 3, narg);
  struct NARRAY* resn;
  GetNArray(res, resn);
  double* lat = reinterpret_cast<double*>(lat_lon->ptr);
  double* lon = lat + n;
  float* line = reinterpret_cast<float*>(resn->ptr);
  float* sample = line + n;
  for(int i = 0; i < n; ++i) {
    MTKt_status status = MtkLatLonToLS(*r, lat[i], lon[i], line + i, 
				       sample + i);
    if(status ==MTK_OUTBOUNDS) {
      line[i] = -1;
      sample[i] = -1;
    } else {
      if(status != MTK_SUCCESS)
	rb_raise(rb_eRuntimeError, "MtkLatLonToLSAry failed");
    }
  }
  return res;
}

/* call-seq:
 *   lat_lon -> [lat, lon]
 *
 * This returns NArray of latitude and longitude (as decimal degrees) for
 * the given MtkDataPlane. This matches the data.
*/

static VALUE mtk_data_plane_lat_lon(VALUE self)
{
  MTKt_MapInfo* r = new MTKt_MapInfo;
  Data_Get_Struct(self, MTKt_MapInfo, r);
  MTKt_DataBuffer latbuf;
  MTKt_DataBuffer lonbuf;
  MTKt_status status = MtkCreateLatLon(*r, &latbuf, &lonbuf);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkCreateLatLon failed");
  VALUE lat = create_narray(NA_DFLOAT, latbuf.data.d, latbuf.nline,
			    latbuf.nsample, (double) 0);
  VALUE lon = create_narray(NA_DFLOAT, lonbuf.data.d, lonbuf.nline,
			    lonbuf.nsample, (double) 0);
  MtkDataBufferFree(&latbuf);
  MtkDataBufferFree(&lonbuf);
  VALUE res = rb_ary_new();
  rb_ary_push(res, lat);
  rb_ary_push(res, lon);
  return res;
}

/*-----------------------------------------------------------*/
/* Routines for MtkCoordinate                                */
/*-----------------------------------------------------------*/

/* call-seq:
 *   block_to_lat_lon(path, resolution, block, line, sample) -> [lat, lon]
 *
 * This returns the latitude and longitude (as decimal degrees) for
 * the given block, line, sample location. Resolution is in meters
 * (e.g., 1100 for 1.1 km data).
*/

static VALUE mtk_coordinate_block_to_lat_lon(VALUE kclass, VALUE path,
		     VALUE resolution, VALUE block, VALUE line, VALUE sample)
{
  double lat, lon;
  MTKt_status status = MtkBlsToLatLon(NUM2INT(path), NUM2INT(resolution),
				      NUM2INT(block), NUM2DBL(line),
				      NUM2DBL(sample), &lat, &lon);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkBlsToLatLon failed");
  VALUE res = rb_ary_new();
  rb_ary_push(res, rb_float_new(lat));
  rb_ary_push(res, rb_float_new(lon));
  return res;
}

/*-----------------------------------------------------------*/
/* Routines for MtkRegion.                                   */
/*-----------------------------------------------------------*/

/* call-seq:
 *   center -> [Latitude, Longitude]]
 *
 * This returns the center coordinates of the region.
 * The latitude and longitude directions are in degrees.
*/

static VALUE mtk_region_center(VALUE self)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  VALUE res;
  res = rb_ary_new();
  rb_ary_push(res, rb_float_new(r->geo.ctr.lat));
  rb_ary_push(res, rb_float_new(r->geo.ctr.lon));
  return res;
}

/* call-seq:
 *   extent -> [Extent latitude diretcion, meters, Extent longitude direction meters]
 *
 * This returns the extent of the region. The extent in
 * the latitude and longitude directions are in meters.
 *
*/

static VALUE mtk_region_extent(VALUE self)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  VALUE res;
  res = rb_ary_new();
  rb_ary_push(res, rb_float_new(r->hextent.xlat * 2));
  rb_ary_push(res, rb_float_new(r->hextent.ylon * 2));
  return res;
}

/* call-seq:
 *   path_list -> Array
 *
 * Returns a list of paths that cross over a given region.
 *
 */

static VALUE mtk_region_path_list(VALUE self)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  int pathcnt;
  int* pathlist;
  MTKt_status status = MtkRegionToPathList(*r, &pathcnt, &pathlist);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkRegionToPathList failed");
  VALUE res = rb_ary_new();
  for(int i = 0; i < pathcnt; ++i)
    rb_ary_push(res, INT2NUM(pathlist[i]));
  free(pathlist);
  return res;
}

/* call-seq:
 *   block_range(path) -> Range
 *
 * Returns a block range that covers the region for the given path.
 *
 */

static VALUE mtk_region_block_range(VALUE self, VALUE path)
{
  MTKt_Region* r;
  Data_Get_Struct(self, MTKt_Region, r);
  int start_block, end_block;
  MTKt_status status = MtkRegionPathToBlockRange(*r, NUM2INT(path), 
						 &start_block, &end_block);
  if(status != MTK_SUCCESS)
    rb_raise(rb_eRuntimeError, "MtkRegionPathToBlockRange failed");
  VALUE rarg[2];
  rarg[0] = INT2NUM(start_block);
  rarg[1] = INT2NUM(end_block);
  VALUE res = rb_class_new_instance(2, rarg, 
	    rb_const_get(rb_cObject, rb_intern("Range")));
  return res;
}

/*-----------------------------------------------------------*/
/* Routines for HdfFile.                                     */
/*-----------------------------------------------------------*/

/* call-seq:
 *   HdfFile.new(filename) -> HdfFile
 *
 * Open the given filename.
 *
 */

static VALUE hdf_init(VALUE self, VALUE name_obj)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    h->open(RSTRING(StringValue(name_obj))->ptr);
    return self;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   close -> HdfFile
 *
 * Close the file. Note that you don't normally need to do this
 * explicitly, garbage collection closes the file automatically.
 *
 */

static VALUE hdf_close(VALUE self)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    h->close();
    return self;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   sds_list -> Array of Strings
 *
 * Return the list of SDSs in the file.
 *
 */

static VALUE sds_list(VALUE self)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    VALUE res = rb_ary_new();
    for(int i = 0; i < (int) h->sds_list().size(); ++i)
      rb_funcall(res, rb_intern("push"), 1, rb_str_new2(h->sds_list()[i].c_str()));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *  sds_size("Field name") => [dim1_size, dim2_size, ...]
 *
 * Return the field size for the given sds.
 */
static VALUE sds_size(VALUE self, VALUE name_obj)
{
  try {
    HdfFile* h;
    Data_Get_Struct(self, HdfFile, h);
    VALUE res = rb_ary_new();
    const std::vector<int32>& sds_dim = 
      h->sds_dim(RSTRING(StringValue(name_obj))->ptr);
    for(int i = 0; i < (int) sds_dim.size(); ++i)
      rb_funcall(res, rb_intern("push"), 1, INT2FIX(sds_dim[i]));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/*-----------------------------------------------------------*/
/* Routines for HdfGrid.                                     */
/*-----------------------------------------------------------*/

/* call-seq:
 *   hdf_file.grid(grid_name) -> HdfGrid
 *
 * An HdfGrid is created through HdfFile#grid
 *
 */

static VALUE grid_init(VALUE self, VALUE hdf_val, VALUE name_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->open(hdf_val, RSTRING(StringValue(name_obj))->ptr);
    return Qnil;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *  field_size("Field name") => [dim1_size, dim2_size, ...]
 *
 * Return the field size for the given grid field.
 */
static VALUE grid_field_size(VALUE self, VALUE name_obj)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    g->populate_field_info(RSTRING(StringValue(name_obj))->ptr);
    VALUE res = rb_ary_new();
    for(int i = 0; i < g->rank(); ++i)
      rb_funcall(res, rb_intern("push"), 1, INT2FIX(g->dim()[i]));
    return res;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *  file => HdfFile
 *
 * Return the HdfFile for the field.
 */
static VALUE grid_file(VALUE self)
{
  try {
    HdfGrid* g;
    Data_Get_Struct(self, HdfGrid, g);
    return g->hdf_obj();
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}


/*-----------------------------------------------------------*/
/* Routines for HdfVdata.                                    */
/*-----------------------------------------------------------*/

/* call-seq:
 *   hdf_file.vdata(vdata_name) -> HdfVdata
 *
 * An HdfVdata is created through HdfFile#vdata
 *
 */

static VALUE vdata_init(VALUE self, VALUE hdf_val, VALUE name_obj)
{
  try {
    HdfVdata* v;
    Data_Get_Struct(self, HdfVdata, v);
    v->open(hdf_val, RSTRING(StringValue(name_obj))->ptr);
    return self;
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}

/* call-seq:
 *   size -> Number records
 *
 * Return the number of records in a vdata.
 */

static VALUE vdata_size(VALUE self)
{
  try {
    HdfVdata* v;
    Data_Get_Struct(self, HdfVdata, v);
    return INT2FIX(v->num_rec());
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "%s", e.what());
  }
}


/*-----------------------------------------------------------*/
/* Initialize classes.                                       */
/*-----------------------------------------------------------*/

VALUE mtk_region_class;
VALUE mtk_file_class;
VALUE mtk_grid_class;
VALUE mtk_field_class;
VALUE mtk_data_plane_class;
VALUE hdf_class;
VALUE grid_class;
VALUE vdata_class;
VALUE mtk_coordinate_class;
void init_mtk_private(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE);
void Init_mtk() {
  mtk_region_class = rb_define_class("MtkRegion", rb_cObject);
  mtk_coordinate_class = rb_define_class("MtkCoordinate", rb_cObject);
  mtk_file_class = rb_define_class("MtkFile", rb_cObject);
  mtk_grid_class = rb_define_class("MtkGrid", rb_cObject);
  mtk_field_class = rb_define_class("MtkField", rb_cObject);
  mtk_data_plane_class = rb_define_class("MtkDataPlane", rb_cObject);
  hdf_class = rb_define_class("HdfFile", rb_cObject);
  grid_class = rb_define_class("HdfGrid", rb_cObject);
  vdata_class = rb_define_class("HdfVdata", rb_cObject);
  rb_define_alloc_func(mtk_data_plane_class, mtk_data_plane_alloc);
  rb_define_alloc_func(mtk_region_class, mtk_region_alloc);
  rb_define_alloc_func(hdf_class, hdf_alloc);
  rb_define_alloc_func(grid_class, grid_alloc);
  rb_define_alloc_func(vdata_class, vdata_alloc);

  init_mtk_private(mtk_region_class, mtk_file_class, mtk_grid_class, 
		   mtk_field_class, mtk_data_plane_class, hdf_class, 
		   grid_class, vdata_class);
/*
  rdoc doesn't recognize the cast down below, so also include as a
  comment w/o the cast.

  rb_define_method(hdf_class, "initialize", hdf_init, 1);
  rb_define_method(hdf_class, "sds_list", sds_list, 0);
  rb_define_method(hdf_class, "sds_size", sds_size, 1);
  rb_define_method(hdf_class, "close", hdf_close, 0);
  rb_define_method(grid_class, "initialize", grid_init, 2);
  rb_define_method(grid_class, "field_size", grid_field_size, 1);
  rb_define_method(grid_class, "file", grid_file, 0);
  rb_define_method(vdata_class, "initialize", vdata_init, 2);
  rb_define_method(vdata_class, "size", vdata_size, 0);
  rb_define_method(mtk_data_plane_class, "lat_lon", mtk_data_plane_lat_lon, 0);
  rb_define_method(mtk_data_plane_class, "lat_lon_to_line_sample", 
  mtk_data_plane_lat_lon_to_line_sample, 1);
  rb_define_method(mtk_region_class, "center", mtk_region_center, 0);
  rb_define_method(mtk_region_class, "extent", mtk_region_extent, 0);
  rb_define_method(mtk_region_class, "path_list", mtk_region_path_list, 0);
  rb_define_method(mtk_region_class, "block_range", mtk_region_block_range, 1);
  rb_define_singleton_method(mtk_coordinate_class, "block_to_lat_lon", mtk_coordinate_block_to_lat_lon, 5);
*/
  rb_define_method(hdf_class, "initialize", (VALUE (*)(...)) hdf_init, 1);
  rb_define_method(hdf_class, "initialize_copy", (VALUE (*)(...)) hdf_copy, 1);
  rb_define_method(mtk_data_plane_class, "initialize_copy", 
		   (VALUE (*)(...)) mtk_data_plane_copy, 1);
  rb_define_method(mtk_data_plane_class, "lat_lon_to_line_sample", 
		   (VALUE (*)(...)) mtk_data_plane_lat_lon_to_line_sample, 1);
  rb_define_method(mtk_region_class, "initialize_copy", 
		   (VALUE (*)(...)) mtk_region_copy, 1);
  rb_define_method(hdf_class, "sds_list", (VALUE (*)(...)) sds_list, 0);
  rb_define_method(hdf_class, "sds_size", (VALUE (*)(...)) sds_size, 1);
  rb_define_method(hdf_class, "close", (VALUE (*)(...)) hdf_close, 0);
  rb_define_method(grid_class, "initialize", (VALUE (*)(...)) grid_init, 2);
  rb_define_method(grid_class, "initialize_copy", (VALUE (*)(...)) grid_copy, 
		   1);
  rb_define_method(grid_class, "field_size", (VALUE (*)(...)) grid_field_size, 1);
  rb_define_method(grid_class, "file", (VALUE (*)(...)) grid_file, 0);
  rb_define_method(vdata_class, "initialize", (VALUE (*)(...)) vdata_init, 2);
  rb_define_method(vdata_class, "initialize_copy", (VALUE (*)(...)) vdata_copy,
		   1);
  rb_define_method(vdata_class, "size", (VALUE (*)(...)) vdata_size, 0);
  rb_define_method(mtk_data_plane_class, "lat_lon", (VALUE (*)(...)) mtk_data_plane_lat_lon, 0);
  rb_define_method(mtk_region_class, "center", (VALUE (*)(...)) mtk_region_center, 0);
  rb_define_method(mtk_region_class, "extent", (VALUE (*)(...)) mtk_region_extent, 0);
  rb_define_method(mtk_region_class, "path_list", (VALUE (*)(...)) mtk_region_path_list, 0);
  rb_define_method(mtk_region_class, "block_range", (VALUE (*)(...)) mtk_region_block_range, 1);
  rb_define_singleton_method(mtk_coordinate_class, "block_to_lat_lon", (VALUE (*)(...)) mtk_coordinate_block_to_lat_lon, 5);
  rb_define_const(mtk_field_class, "CHAR8", INT2NUM(MTKe_char8));
  rb_define_const(mtk_field_class, "UCHAR8", INT2NUM(MTKe_uchar8));
  rb_define_const(mtk_field_class, "INT16", INT2NUM(MTKe_int16));
  rb_define_const(mtk_field_class, "UINT16", INT2NUM(MTKe_uint16));
  rb_define_const(mtk_field_class, "INT32", INT2NUM(MTKe_int32));
  rb_define_const(mtk_field_class, "UINT32", INT2NUM(MTKe_uint32));
  rb_define_const(mtk_field_class, "INT64", INT2NUM(MTKe_int64));
  rb_define_const(mtk_field_class, "UINT64", INT2NUM(MTKe_uint64));
  rb_define_const(mtk_field_class, "FLOAT", INT2NUM(MTKe_float));
  rb_define_const(mtk_field_class, "DOUBLE", INT2NUM(MTKe_double));
  rb_define_const(mtk_file_class, "AGP", INT2NUM(MTK_AGP));
  rb_define_const(mtk_file_class, "GP_GMP", INT2NUM(MTK_GP_GMP));
  rb_define_const(mtk_file_class, "GRP_RCCM_GM", INT2NUM(MTK_GRP_RCCM_GM));
  rb_define_const(mtk_file_class, "GRP_ELLIPSOID_GM", INT2NUM(MTK_GRP_ELLIPSOID_GM));
  rb_define_const(mtk_file_class, "GRP_TERRAIN_GM", INT2NUM(MTK_GRP_TERRAIN_GM));
  rb_define_const(mtk_file_class, "GRP_ELLIPSOID_LM", INT2NUM(MTK_GRP_ELLIPSOID_LM));
  rb_define_const(mtk_file_class, "GRP_TERRAIN_LM", INT2NUM(MTK_GRP_TERRAIN_LM));
  rb_define_const(mtk_file_class, "AS_AEROSOL", INT2NUM(MTK_AS_AEROSOL));
  rb_define_const(mtk_file_class, "AS_LAND", INT2NUM(MTK_AS_LAND));
  rb_define_const(mtk_file_class, "TC_ALBEDO", INT2NUM(MTK_TC_ALBEDO));
  rb_define_const(mtk_file_class, "TC_CLASSIFIERS", INT2NUM(MTK_TC_CLASSIFIERS));
  rb_define_const(mtk_file_class, "TC_STEREO", INT2NUM(MTK_TC_STEREO));
  rb_define_const(mtk_file_class, "CONVENTIONAL", INT2NUM(MTK_CONVENTIONAL));
  rb_define_const(mtk_file_class, "UNKNOWN", INT2NUM(MTK_UNKNOWN));
}

