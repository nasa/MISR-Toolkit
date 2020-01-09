/*===========================================================================
=                                                                           =
=                          MtkReadBlockRange_test                           =
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
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
				/* Data buffer structure */
  char filename[80];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS gridname */
  char fieldname[80];		/* HDF-EOS fieldname */
  int startblock;		/* Start block to read */
  int endblock;			/* End block to read */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkReadBlockRange");

  /* Normal test call */
  startblock = 26;
  endblock = 40;
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");

  status = MtkReadBlockRange(filename, gridname, fieldname,
			     startblock, endblock, &dbuf);
  if (status == MTK_SUCCESS &&
      dbuf.data.u16[0][74][262] == 357 &&
      dbuf.data.u16[14][74][262] == 256) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree3D(&dbuf);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(NULL, gridname, fieldname, startblock, endblock,
			     &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, NULL, fieldname, startblock, endblock,
			     &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, NULL, startblock, endblock,
			     &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, 0, endblock,
			     &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, startblock, 0,
			     &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, 40, 26, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_LAND_P039_O002467_F08_23.b056-070.nc");
  {
    int startblock = 58;
    int endblock = 61;
    strcpy(gridname, "1.1_KM_PRODUCTS");
    strcpy(fieldname, "Directional_Hemispherical_Reflectance[2]");
    
    status = MtkReadBlockRange(filename, gridname, fieldname,
                               startblock, endblock, &dbuf);
    if (status == MTK_SUCCESS &&
        dbuf.data.u8[1][64][256] == 59 &&  // block 59, line 63, sample 256; file offset 448, 416
        dbuf.data.u8[1][63][255] == 53) {  // block 59, line 63, sample 256; file offset 448, 416
      MTK_PRINT_STATUS(cn,".");
      MtkDataBufferFree3D(&dbuf);
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  }

  status = MtkReadBlockRange(filename, NULL, fieldname, startblock, endblock,
			     &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, NULL, startblock, endblock,
			     &dbuf);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, 0, endblock,
			     &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, startblock, 0,
			     &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkReadBlockRange(filename, gridname, fieldname, 40, 26, &dbuf);
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
