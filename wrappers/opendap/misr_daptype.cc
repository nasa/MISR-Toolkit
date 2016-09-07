/*===========================================================================
=                                                                           =
=                             misr_daptype                                  =
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
#include "DDS.h"
#include "misr_types.h"
#include "MisrToolkit.h"

static const char *error_msg[] = MTK_ERR_DESC;

/* ------------------------------------------------------------------------ */
/*  Create DAP BaseType object based on MisrToolkit datatype	       	    */
/* ------------------------------------------------------------------------ */

BaseType *misr_daptype(const char *filename, const char *gridname,
		       const char *fieldname)
{
  MTKt_status status;
  MTKt_DataType datatype;

  status = MtkFileGridFieldToDataType(filename, gridname, fieldname,
				      &datatype);
  if (status != MTK_SUCCESS)
    throw Error("MtkFileGridFieldToDataType(): "+string(error_msg[status]));

  switch(datatype) {

  case MTKe_char8:
  case MTKe_uchar8:
  case MTKe_int8:
  case MTKe_uint8:
    return NewByte(string(fieldname));

  case MTKe_int16:
    return NewInt16(string(fieldname));

  case MTKe_uint16:
    return NewUInt16(string(fieldname));

  case MTKe_int32:
    return NewUInt32(string(fieldname));

  case MTKe_uint32:
    return NewUInt32(string(fieldname));

  case MTKe_int64:
  case MTKe_uint64:
    return 0;

  case MTKe_float:
    return NewFloat32(string(fieldname));

  case MTKe_double:
    return NewFloat64(string(fieldname));

  default:
    return 0;
  }
}
