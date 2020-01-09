/*===========================================================================
=                                                                           =
=                            MtkReadBlock_test                              =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReadData.h"
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
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int block;			/* Block to read */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkReadBlock");

  /* Normal test call */
  block = 26;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  status = MtkReadBlock(filename, gridname, fieldname, block, &dbuf);
  if (status == MTK_SUCCESS &&
      dbuf.data.u16[74][262] == 357) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(NULL, gridname, fieldname, block, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(filename, NULL, fieldname, block, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(filename, gridname, NULL, block, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(filename, gridname, fieldname, 0, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  strcpy(gridname, "1.1_KM_PRODUCTS");
  strcpy(fieldname, "Directional_Hemispherical_Reflectance[2]");

  {
    int block = 59;
    status = MtkReadBlock(filename, gridname, fieldname, block, &dbuf);
    if (status == MTK_SUCCESS &&
        dbuf.data.u8[64][256] == 59 &&  // offset 448, 416
        dbuf.data.u8[63][255] == 53) {  // offset 448, 416
      MTK_PRINT_STATUS(cn,".");
      MtkDataBufferFree(&dbuf);
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  status = MtkReadBlock(filename, NULL, fieldname, block, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(filename, gridname, NULL, block, &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlock(filename, gridname, fieldname, 0, &dbuf);
  if (status == MTK_OUTBOUNDS) {
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
