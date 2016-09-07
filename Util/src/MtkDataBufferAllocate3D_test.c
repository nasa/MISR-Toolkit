/*===========================================================================
=                                                                           =
=                       MtkDataBufferAllocate3D_test                        =
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
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean good = MTK_TRUE; /* Good data flag */
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT;
				/* Data buffer structure */
  int b;			/* block index */
  int l;			/* Line index */
  int s;			/* Sample index */
  int nblock;			/* Number of blocks */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int datatype;			/* Data type */
  MTKt_uint16 data;		/* Data element */
  MTKt_float data2;		/* Data element */
  MTKt_double data3;		/* Data element */
  MTKt_int8 data4;		/* Data element */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDataBufferAllocate3D");

  /* Normal test call */
  nblock = 3;
  nline = 5;
  nsample = 10;
  datatype = MTKe_uint16;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_SUCCESS) {
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data = (b+1) * l * s;
	  dbuf.data.u16[b][l][s] = data;
	}
      }
    }
    good = MTK_TRUE;
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data = (b+1) * l * s;
	  if (dbuf.data.u16[b][l][s] != data){
	    good = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree3D(&dbuf);
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nblock = 3;
  nline = 10;
  nsample = 5;
  datatype = MTKe_float;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_SUCCESS) {
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data2 = 1.0 / ((b+1) * (l+1) * (s+1));
	  dbuf.data.f[b][l][s] = data2;
	}
      }
    }
    good = MTK_TRUE;
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data2 = 1.0 / ((b+1) * (l+1) * (s+1));
	  if (fabsf(dbuf.data.f[b][l][s] - data2) > .000001){
	    good = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree3D(&dbuf);
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
   } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nblock = 3;
  nline = 13;
  nsample = 12;
  datatype = MTKe_double;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_SUCCESS) {
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data3 = 1.0 / ((b+1) * (l+1) * (s+1));
	  dbuf.data.d[b][l][s] = data3;
	}
      }
    }
    good = MTK_TRUE;
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data3 = 1.0 / ((b+1) * (l+1) * (s+1));
	  if (fabs(dbuf.data.d[b][l][s] - data3) > .000001){
	    good = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree3D(&dbuf);
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nblock = 3;
  nline = 5;
  nsample = 3;
  datatype = MTKe_int8;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_SUCCESS) {
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data4 = b * l * -s;
	  dbuf.data.i8[b][l][s] = data4;
	}
      }
    }
    good = MTK_TRUE;
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  data4 = b * l * -s;
	  if (dbuf.data.i8[b][l][s] != data4){
	    good = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree3D(&dbuf);
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
   } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  nblock = 3;
  nline = 1;
  nsample = 1;
  datatype = MTKe_float;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_SUCCESS) {
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  dbuf.data.f[b][l][s] = 0.0;
	}
      }
    }
    good = MTK_TRUE;
    for (b = 0; b < dbuf.nblock; b++) {
      for (l = 0; l < dbuf.nline; l++) {
	for (s = 0; s < dbuf.nsample; s++) {
	  if (dbuf.data.f[b][l][s] != 0.0){
	    good = MTK_FALSE;
	  }
	}
      }
    }
    MtkDataBufferFree3D(&dbuf);
    if (good) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  nblock = -3;
  nline = 5;
  nsample = 3;
  datatype = MTKe_int8;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  nblock = 3;
  nline = -5;
  nsample = 3;
  datatype = MTKe_int8;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  nblock = 3;
  nline = 5;
  nsample = -3;
  datatype = MTKe_int8;

  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDataBufferAllocate3D(nblock, nline, nsample, datatype, NULL);
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
