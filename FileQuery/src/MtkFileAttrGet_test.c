/*===========================================================================
=                                                                           =
=                            MtkFileAttrGet_test                            =
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
  char attrname[80];		/* HDF-EOS attrname */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkFileAttrGet");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  strcpy(attrname, "Camera");

  status = MtkFileAttrGet(filename, attrname, &dbuf);
  if (status == MTK_SUCCESS && dbuf.data.i32[0][0] == 6) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test calls */
  strcpy(filename, "../Mtk_testdata/in/abcd.hdf");
  strcpy(attrname, "Camera");

  status = MtkFileAttrGet(filename, attrname, &dbuf);
  if (status == MTK_HDF_SDSTART_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(attrname, "");

  status = MtkFileAttrGet(filename, attrname, &dbuf);
  if (status == MTK_HDF_SDFINDATTR_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Argument Checks */
  status = MtkFileAttrGet(NULL, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrGet(filename, NULL, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrGet(filename, attrname, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(attrname, "Ocean_blocks.count");

  status = MtkFileAttrGet(filename, attrname, &dbuf);
  if (status == MTK_SUCCESS && dbuf.data.i32[0][0] == 87) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  strcpy(attrname, "");

  status = MtkFileAttrGet(filename, attrname, &dbuf);
  if (status == MTK_NETCDF_READ_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  
  /* Argument Checks */
  status = MtkFileAttrGet(NULL, attrname, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrGet(filename, NULL, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrGet(filename, attrname, NULL);
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
