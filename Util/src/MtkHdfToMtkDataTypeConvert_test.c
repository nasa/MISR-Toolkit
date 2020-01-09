/*===========================================================================
=                                                                           =
=                     MtkHdfToMtkDataTypeConvert_test                       =
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
#include <stdio.h>
#include <mfhdf.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataType datatype; 	/* Mtk datatype */
  int32 hdf_datatype;		/* HDF datatype */
  nc_type nc_datatype;		/* HDF datatype */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkHdfToMtkDataTypeConvert");

  /* Normal test call */
  hdf_datatype = DFNT_CHAR8;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_char8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_UCHAR8;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uchar8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_INT8;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_UINT8;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_INT16;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int16) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_UINT16;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint16) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_INT32;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int32) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_UINT32;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint32) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_INT64;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int64) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_UINT64;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint64) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_FLOAT32;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_float) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  hdf_datatype = DFNT_FLOAT64;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_double) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  hdf_datatype = 5023;
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, &datatype);
  if (status == MTK_FAILURE) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkHdfToMtkDataTypeConvert(hdf_datatype, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_CHAR;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_char8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_BYTE;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_UBYTE;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint8) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_SHORT;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int16) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_USHORT;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint16) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_INT;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int32) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_UINT;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint32) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_INT64;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_int64) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_UINT64;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_uint64) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_FLOAT;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_float) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nc_datatype = NC_DOUBLE;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_SUCCESS &&
      datatype == MTKe_double) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  nc_datatype = 5023;
  status = MtkNcToMtkDataTypeConvert(nc_datatype, &datatype);
  if (status == MTK_FAILURE) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkNcToMtkDataTypeConvert(nc_datatype, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
