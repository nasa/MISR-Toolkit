/*===========================================================================
=                                                                           =
=                          MtkDataBufferFree_test                           =
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
#include <stdlib.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				/* Data buffer structure */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int datatype;			/* Data type */
  void *dataptr;		/* Data pointer */
  int datasize[] = MTKd_DataSize;
				/* Data sizes by data type */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDataBufferFree");

  /* Normal test call */
  nline = 5;
  nsample = 10;
  datatype = MTKe_uint16;

  status = MtkDataBufferAllocate(nline, nsample, datatype, &dbuf);

  if (status == MTK_SUCCESS) {

    status = MtkDataBufferFree(&dbuf);
    if (status == MTK_SUCCESS) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nline = 5;
  nsample = 10;
  datatype = MTKe_uint16;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);

  if (status == MTK_SUCCESS) {

    status = MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  status = MtkDataBufferFree(NULL);
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
