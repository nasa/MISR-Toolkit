/*===========================================================================
=                                                                           =
=                               misr_types                                  =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include <string>
#include <iostream>
#include <sstream>
#include "misr_types.h"
#include "dods-datatypes.h"
#include "InternalErr.h"

#include "MisrToolkit.h"

static const char *error_msg[] = MTK_ERR_DESC;

bool misr_readarray(const string &filename, MISRArray *bt);

/* ------------------------------------------------------------------------ */
/* MISRByte								    */
/* ------------------------------------------------------------------------ */

Byte *
NewByte(const string &n)
{
    return new MISRByte(n);
}

MISRByte::MISRByte(const string &n) : Byte(n)
{
}

BaseType *
MISRByte::ptr_duplicate()
{
    return new MISRByte(*this);
}

bool
MISRByte::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRUInt16								    */
/* ------------------------------------------------------------------------ */

UInt16 *
NewUInt16(const string &n)
{
    return new MISRUInt16(n);
}

MISRUInt16::MISRUInt16(const string &n) : UInt16(n)
{
}

BaseType *
MISRUInt16::ptr_duplicate()
{
    return new MISRUInt16(*this);
}

bool
MISRUInt16::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRInt16								    */
/* ------------------------------------------------------------------------ */

Int16 *
NewInt16(const string &n)
{
    return new MISRInt16(n);
}

MISRInt16::MISRInt16(const string &n) : Int16(n)
{
}

BaseType *
MISRInt16::ptr_duplicate()
{
    return new MISRInt16(*this);
}

bool
MISRInt16::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRUInt32								    */
/* ------------------------------------------------------------------------ */

UInt32 *
NewUInt32(const string &n)
{
    return new MISRUInt32(n);
}

MISRUInt32::MISRUInt32(const string &n) : UInt32(n)
{
}

BaseType *
MISRUInt32::ptr_duplicate()
{
    return new MISRUInt32(*this);
}

bool
MISRUInt32::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRInt32								    */
/* ------------------------------------------------------------------------ */

Int32 *
NewInt32(const string &n)
{
    return new MISRInt32(n);
}

MISRInt32::MISRInt32(const string &n) : Int32(n)
{
}

BaseType *
MISRInt32::ptr_duplicate()
{
    return new MISRInt32(*this);
}

bool
MISRInt32::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRFloat32								    */
/* ------------------------------------------------------------------------ */

Float32 *
NewFloat32(const string &n)
{
    return new MISRFloat32(n);
}

MISRFloat32::MISRFloat32(const string &n) : Float32(n)
{
}

BaseType *
MISRFloat32::ptr_duplicate()
{
    return new MISRFloat32(*this);
}

bool
MISRFloat32::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRFloat64								    */
/* ------------------------------------------------------------------------ */

Float64 *
NewFloat64(const string &n)
{
    return new MISRFloat64(n);
}

MISRFloat64::MISRFloat64(const string &n) : Float64(n)
{
}

BaseType *
MISRFloat64::ptr_duplicate()
{
    return new MISRFloat64(*this);
}

bool
MISRFloat64::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRStr								    */
/* ------------------------------------------------------------------------ */

Str *
NewStr(const string &n)
{
    return new MISRStr(n);
}

MISRStr::MISRStr(const string &n) : Str(n)
{
}

BaseType *
MISRStr::ptr_duplicate()
{
    return new MISRStr(*this);
}

bool
MISRStr::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRUrl								    */
/* ------------------------------------------------------------------------ */

Url *
NewUrl(const string &n)
{
    return new MISRUrl(n);
}

MISRUrl::MISRUrl(const string &n) : Url(n)
{
}

BaseType *
MISRUrl::ptr_duplicate()
{
    return new MISRUrl(*this);
}

bool
MISRUrl::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRArray								    */
/* ------------------------------------------------------------------------ */

Array *
NewArray(const string &n)
{
  return new MISRArray(n);
}

MISRArray::MISRArray(const string &n) : Array(n)
{
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
  this->global_mapinfo = mapinfo;
}

void
MISRArray::set_mapinfo(MTKt_MapInfo mapinfo)
{
  this->global_mapinfo = mapinfo;
}

BaseType *
MISRArray::ptr_duplicate()
{
  return new MISRArray(*this);
}

bool
MISRArray::read(const string &filename)
{
  MTKt_Region region = MTKT_REGION_INIT;
  MTKt_DataBuffer databuffer = MTKT_DATABUFFER_INIT;
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
  MTKt_status status;

  if (read_p()) // Nothing to do
    return false;

  string gridname = this->get_parent()->name();
  string fieldname = this->name();

#ifdef DEBUG
  std::cerr << "Mtk[MISRArray::read()]: ";
  std::cerr << filename << ":";
  std::cerr << gridname << ":";
  std::cerr << fieldname << std::endl;
#endif

  int i=0;
  int start[6], stride[6], stop[6], edge[6];

  for (Array::Dim_iter p = dim_begin(); p != dim_end(); ++p) {
    start[i] = dimension_start(p,true);
    stride[i] = dimension_stride(p,true);
    stop[i] = dimension_stop(p,true);
    edge[i] = (int)((stop[i] - start[i])/stride[i]) + 1;
#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << i << ") ";
    std::cerr << "(" << start[i] << ":" << stride[i] << ":" << stop[i] << ") ";
    std::cerr << "(" << edge[i] << ")" << std::endl;
#endif
    i++;
  }

  double ulc_som_x, ulc_som_y;
  double lrc_som_x, lrc_som_y;

  status = MtkLSToSomXY(global_mapinfo, start[0], start[1],
			&ulc_som_x, &ulc_som_y);
  if (status != MTK_SUCCESS)
    throw Error("MtkLSToSomXY(): "+string(error_msg[status]));

  status = MtkLSToSomXY(global_mapinfo, stop[0], stop[1],
			&lrc_som_x, &lrc_som_y);
  if (status != MTK_SUCCESS)
    throw Error("MtkLSToSomXY(): "+string(error_msg[status]));

#ifdef DEBUG
  std::cerr << "Mtk[MISRArray::read()]: ";
  std::cerr << "ULC: (" << ulc_som_x << "," << ulc_som_y << ") " << std::endl;
  std::cerr << "Mtk[MISRArray::read()]: ";
  std::cerr << "LRC: (" << lrc_som_x << "," << lrc_som_y << ") " << std::endl;
#endif

  status = MtkSetRegionByPathSomUlcLrc(global_mapinfo.path,
				       ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y, &region);
  if (status != MTK_SUCCESS)
    throw Error("MtkSetRegionByPathSomUlcLrc(): "+string(error_msg[status]));

  /* -------------------------------------------------------------------- */
  /* Determine if the field is a computable lat/lon field or a real field */
  /* -------------------------------------------------------------------- */

  if (fieldname == gridname+string("Latitude") ||
      fieldname == gridname+string("Longitude")) {

    /* ------------------------------------------------------------------ */
    /* Compute lat/lon fields                                             */
    /* ------------------------------------------------------------------ */

    MTKt_DataBuffer latbuffer = MTKT_DATABUFFER_INIT;
    MTKt_DataBuffer lonbuffer = MTKT_DATABUFFER_INIT;

    status = MtkSnapToGrid(global_mapinfo.path, global_mapinfo.resolution,
			   region, &mapinfo);
    if (status != MTK_SUCCESS)
      throw Error("MtkSnapToGrid(): "+string(error_msg[status]));

#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "MtkCreateLatLon started" << std::endl;
#endif

    status = MtkCreateLatLon(mapinfo, &latbuffer, &lonbuffer);
    if (status != MTK_SUCCESS) {
#ifdef DEBUG
      std::cerr << "Mtk[MtkCreateLatLon()]: "+string(error_msg[status]) << endl;
#endif
      throw Error("MtkCreateLatLon(): "+string(error_msg[status]));
    }

#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "MtkCreateLatLon finished" << std::endl;
#endif

    if (fieldname == gridname+string("Latitude")) {

      int Tcount = 0;
      int Len = ((mapinfo.nline/stride[0])+1) *
	((mapinfo.nsample/stride[1])+1);

      dods_float64 *destbuf = new dods_float64 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_float64 *)latbuffer.dataptr)[lp + sample];
	}
      }
      val2buf((void *)destbuf);
    }

    if (fieldname == gridname+string("Longitude")) {

      int Tcount = 0;
      int Len = ((mapinfo.nline/stride[0])+1) *
	((mapinfo.nsample/stride[1])+1);

      dods_float64 *destbuf = new dods_float64 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_float64 *)lonbuffer.dataptr)[lp + sample];
	}
      }
      val2buf((void *)destbuf);
    }

    status = MtkDataBufferFree(&latbuffer);
    if (status != MTK_SUCCESS)
      throw Error("MtkDataBufferFree(): "+string(error_msg[status]));

    status = MtkDataBufferFree(&lonbuffer);
    if (status != MTK_SUCCESS)
      throw Error("MtkDataBufferFree(): "+string(error_msg[status]));

  } else {

    /* ------------------------------------------------------------------ */
    /* Read the field                                                     */
    /* ------------------------------------------------------------------ */

    int dimcnt;
    char **dimlist;
    int *dimsize;
    std::ostringstream dimstr;

    status = MtkFileGridFieldToDimList(filename.c_str(), gridname.c_str(),
				       fieldname.c_str(), &dimcnt,
				       &dimlist, &dimsize);
    if (status != MTK_SUCCESS) {
      throw Error("MtkFieldGridFieldToDimList(): "+string(error_msg[status]));
    }
    MtkStringListFree(dimcnt, &dimlist);
    free(dimsize);

    for (int i = 0; i < dimcnt; i++) {
      dimstr << start[i];
      fieldname += "[" + dimstr.str() + "]";
    }
    
#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "MtkReadData started ";
    std::cerr << fieldname << std::endl;
#endif

    status = MtkReadData(filename.c_str(), gridname.c_str(), fieldname.c_str(),
			 region, &databuffer, &mapinfo);
    if (status != MTK_SUCCESS) {
#ifdef DEBUG
      std::cerr << "Mtk[MtkReadData()]: "+string(error_msg[status]) << endl;
#endif
      throw Error("MtkReadData(): "+string(error_msg[status]));
    }

#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "MtkReadData finished" << std::endl;
#endif

#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "LS: (" << databuffer.nline << ",";
    std::cerr << databuffer.nsample << ") " << std::endl;
#endif

    int Tcount = 0;
    int Len = (mapinfo.nline/stride[0]) * (mapinfo.nsample/stride[1]);

#ifdef DEBUG
    std::cerr << "Mtk[MISRArray::read()]: ";
    std::cerr << "Len: " << Len << std::endl;
#endif
    switch (databuffer.datatype) {

    case MTKe_char8: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_char8" << std::endl;
#endif
      dods_byte *destbuf = new dods_byte [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_byte *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
    case MTKe_uchar8: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_uchar8" << std::endl;
#endif
      dods_byte *destbuf = new dods_byte [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_byte *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
      
    case MTKe_int8: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_int8" << std::endl;
#endif
      dods_byte *destbuf = new dods_byte [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_byte *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
    case MTKe_uint8: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_uint8" << std::endl;
#endif
      dods_byte *destbuf = new dods_byte [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_byte *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
      
    case MTKe_int16: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_int16" << std::endl;
#endif
      dods_int16 *destbuf = new dods_int16 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_int16 *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
    case MTKe_uint16: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_uint16" << std::endl;
#endif
      dods_uint16 *destbuf = new dods_uint16 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_uint16 *)databuffer.dataptr)[lp + sample];
	}
      }
      val2buf((void *)destbuf);
      break;
    }
      
    case MTKe_int32: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_int32" << std::endl;
#endif
      dods_int32 *destbuf = new dods_int32 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_int32 *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
    case MTKe_uint32: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_uint32" << std::endl;
#endif
      dods_uint32 *destbuf = new dods_uint32 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_uint32 *)databuffer.dataptr)[lp + sample];
	}
      }
      val2buf((void *)destbuf);
      break;
    }
      
    case MTKe_int64: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_int64" << std::endl;
#endif
      throw Error("Signed 64-bit integer not supported in opendap");
      break;
    }
    case MTKe_uint64: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_uint64" << std::endl;
#endif
      throw Error("Unsigned 64-bit integer not supported in opendap");
      break;
    }
      
    case MTKe_float: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_float" << std::endl;
#endif
      dods_float32 *destbuf = new dods_float32 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_float32 *)databuffer.dataptr)[lp + sample];
	}
	val2buf((void *)destbuf);
      }
      break;
    }
    case MTKe_double: {
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: MTKe_double" << std::endl;
#endif
      dods_float64 *destbuf = new dods_float64 [Len];
      for (int line = 0; line < mapinfo.nline; line +=stride[0]) {	  
	int lp = line * mapinfo.nsample;
	for(int sample = 0; sample < mapinfo.nsample; sample+=stride[1]){
	  *(destbuf+Tcount++) =
	    ((dods_float64 *)databuffer.dataptr)[lp + sample];
	}
      }
      val2buf((void *)destbuf);
      break;
    }
    default:
#ifdef DEBUG
      std::cerr << "Mtk[MISRArray::read()]: Unsupported datatype" << std::endl;
#endif
      throw Error("Unsupported datatype");
    }
  } /* End of else portion of lat/lon grid/field test */

  status = MtkDataBufferFree(&databuffer);
  if (status != MTK_SUCCESS)
    throw Error("MtkDataBufferFree(): "+string(error_msg[status]));
  
#ifdef DEBUG
  std::cerr << "Mtk[MISRArray::read()]: Done" << std::endl;
#endif

  set_read_p(true);
  return false;
}

/* ------------------------------------------------------------------------ */
/* MISRStructure						      	    */
/* ------------------------------------------------------------------------ */

Structure *
NewStructure(const string &n)
{
    return new MISRStructure(n);
}

MISRStructure::MISRStructure(const string &n) : Structure(n)
{
}

BaseType *
MISRStructure::ptr_duplicate()
{
    return new MISRStructure(*this);
}

bool
MISRStructure::read(const string &filename)
{
  string gridname = this->name();

#ifdef DEBUG
  std::cerr << "Mtk[MISRStructure::read()]: ";
  std::cerr << filename << ":";
  std::cerr << gridname << std::endl;
#endif

  return false;
}

/* ------------------------------------------------------------------------ */
/* MISRSequence								    */
/* ------------------------------------------------------------------------ */

Sequence *
NewSequence(const string &n)
{
    return new MISRSequence(n);
}

MISRSequence::MISRSequence(const string &n) : Sequence(n)
{
}

BaseType *
MISRSequence::ptr_duplicate()
{
    return new MISRSequence(*this);
}

bool
MISRSequence::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}

/* ------------------------------------------------------------------------ */
/* MISRGrid								    */
/* ------------------------------------------------------------------------ */

Grid *
NewGrid(const string &n)
{
    return new MISRGrid(n);
}

MISRGrid::MISRGrid(const string &n) : Grid(n)
{
}

BaseType *
MISRGrid::ptr_duplicate()
{
    return new MISRGrid(*this);
}

bool
MISRGrid::read(const string &)
{
    throw InternalErr(__FILE__, __LINE__, "Unimplemented read method called.");
}
