/*===========================================================================
=                                                                           =
=                         MtkDataBufferFree3D_test                          =
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

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
				/* Data buffer structure */
  int nblock;			/* Number of blocks */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int datatype;			/* Data type */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDataBufferFree3D");

  /* Normal test call */
  nblock = 3;
  nline = 5;
  nsample = 10;
  datatype = MTKe_uint16;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);

  if (status == MTK_SUCCESS) {

    status = MtkDataBufferFree3D(&dbuf);
    if (status == MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDataBufferFree3D(NULL);
  if (status == MTK_SUCCESS) {
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
