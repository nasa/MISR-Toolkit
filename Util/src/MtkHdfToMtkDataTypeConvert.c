/*===========================================================================
=                                                                           =
=                        MtkHdfToMtkDataTypeConvert                         =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUtil.h"
#include <mfhdf.h>

/** \brief Convert HDF data type to MISR Toolkit data type
 *
 *  \return MISR Toolkit data type
 */

MTKt_status MtkHdfToMtkDataTypeConvert( int32 hdf_datatype,
				        MTKt_DataType *datatype ) {

  if (datatype == NULL)
    return MTK_NULLPTR;

  switch (hdf_datatype) {
  case DFNT_CHAR8: *datatype = MTKe_char8;
    break;
  case DFNT_UCHAR8: *datatype = MTKe_uchar8;
    break;
  case DFNT_INT8: *datatype = MTKe_int8;
    break;
  case DFNT_UINT8: *datatype = MTKe_uint8;
    break;
  case DFNT_INT16: *datatype = MTKe_int16;
    break;
  case DFNT_UINT16: *datatype = MTKe_uint16;
    break;
  case DFNT_INT32: *datatype = MTKe_int32;
    break;
  case DFNT_UINT32: *datatype = MTKe_uint32;
    break;
  case DFNT_INT64: *datatype = MTKe_int64;
    break;
  case DFNT_UINT64: *datatype = MTKe_uint64;
    break;
  case DFNT_FLOAT32: *datatype = MTKe_float;
    break;
  case DFNT_FLOAT64: *datatype = MTKe_double;
    break;
  default:
    return MTK_FAILURE;
  }

  return MTK_SUCCESS;
}

MTKt_status MtkNcToMtkDataTypeConvert( nc_type nc_datatype,
                                       MTKt_DataType *datatype ) {

  if (datatype == NULL)
    return MTK_NULLPTR;

  switch (nc_datatype) {
  case NC_CHAR: *datatype = MTKe_char8;
    break;
  case NC_UBYTE: *datatype = MTKe_uint8;
    break;
  case NC_BYTE: *datatype = MTKe_int8;
    break;
  case NC_SHORT: *datatype = MTKe_int16;
    break;
  case NC_USHORT: *datatype = MTKe_uint16;
    break;
  case NC_INT: *datatype = MTKe_int32;
    break;
  case NC_UINT: *datatype = MTKe_uint32;
    break;
  case NC_INT64: *datatype = MTKe_int64;
    break;
  case NC_UINT64: *datatype = MTKe_uint64;
    break;
  case NC_FLOAT: *datatype = MTKe_float;
    break;
  case NC_DOUBLE: *datatype = MTKe_double;
    break;
  default:
    return MTK_FAILURE;
  }

  return MTK_SUCCESS;
}
