/*===========================================================================
=                                                                           =
=                            MtkFieldAttrGet_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				/* Data buffer structure */
  char filename[80];		/* HDF-EOS filename */
  char fieldname[80]; /* HDF SDS Field Name */
  char attrname[80];		/* HDF-EOS attrname */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFieldAttrGet");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(fieldname, "CloudMotionCrossTrack");
  strcpy(attrname, "_FillValue");

  status = MtkFieldAttrGet(filename, fieldname, attrname, &dbuf);
  if (status == MTK_SUCCESS && dbuf.data.i16[0][0] == -22222) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test calls */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
  

  status = MtkFieldAttrGet(filename, fieldname, attrname, &dbuf);
  if (status == MTK_HDFEOS_GDOPEN_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  strcpy(attrname, "");

  status = MtkFieldAttrGet(filename, fieldname, attrname, &dbuf);
  if (status == MTK_HDF_SDFINDATTR_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Argument Checks */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_TC_CLOUD_P110_O074017_F01_0001.hdf");
  status = MtkFieldAttrGet(NULL, fieldname, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrGet(filename, NULL, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrGet(filename, fieldname, NULL, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrGet(filename, fieldname, attrname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(fieldname, "Latitude");
  strcpy(attrname, "_FillValue");

  status = MtkFieldAttrGet(filename, fieldname, attrname, &dbuf);
  if (status == MTK_SUCCESS && dbuf.data.f[0][0] == -9999) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(attrname, "");

  status = MtkFieldAttrGet(filename, fieldname, attrname, &dbuf);
  if (status == MTK_NETCDF_READ_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Argument Checks */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  status = MtkFieldAttrGet(NULL, fieldname, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  status = MtkFieldAttrGet(filename, NULL, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrGet(filename, fieldname, NULL, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFieldAttrGet(filename, fieldname, attrname, NULL);
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
