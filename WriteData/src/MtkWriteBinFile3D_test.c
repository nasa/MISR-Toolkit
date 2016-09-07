/*===========================================================================
=                                                                           =
=                          MtkWriteBinFile3D_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrWriteData.h"
#include "MisrSetRegion.h"
#include "MisrReadData.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main() {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
                                /* Data buffer structure */
  char filename[300];		/* HDF-EOS filename */
  char gridname[80];		/* HDF-EOS fieldname */
  char fieldname[80];		/* HDF-EOS fieldname */
  char outfilename[300];	/* Binary filename */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkWriteBinFile3D");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AGP_P037_F01_24.hdf");
  strcpy(gridname, "Standard");
  strcpy(fieldname, "AveSceneElev");
  strcpy(outfilename, "../Mtk_testdata/out/MISR_AM1_AGP_P037_F01_24_blocks");

  MtkReadBlockRange(filename, gridname, fieldname, 50, 60, &dbuf);

  status = MtkWriteBinFile3D(outfilename, dbuf);
  if (status == MTK_SUCCESS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  MtkDataBufferFree3D(&dbuf);

  status = MtkWriteBinFile3D(NULL, dbuf);
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
