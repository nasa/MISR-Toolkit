/*===========================================================================
=                                                                           =
=                          MtkDataBufferImport_test                         =
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
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean good = MTK_TRUE; /* Good data flag */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;
				/* Data buffer structure */
  void *dataptr;		/* Data pointer to buffer to by imported */
  int datasize[] = MTKd_DataSize;
				/* Data sizes by data type */
  int l;			/* Line index */
  int s;			/* Sample index */
  int lp;			/* Line stride */
  int nline;			/* Number of lines */
  int nsample;			/* Number of samples */
  int datatype;			/* Data type */
  MTKt_uint16 data;		/* Data element */
  MTKt_float data2;		/* Data element */
  MTKt_double data3;		/* Data element */
  MTKt_int8 data4;		/* Data element */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkDataBufferImport");

  /* Normal test call */
  nline = 5;
  nsample = 10;
  datatype = MTKe_uint16;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  for (l = 0; l < nline; l++) {
    lp = l * nsample;
    for (s = 0; s < nsample; s++) {
      data = l * s;
      ((unsigned short *)dataptr)[lp + s] = data;
    }
  }

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_SUCCESS) {
    good = MTK_TRUE;
    for (l = 0; l < dbuf.nline; l++) {
      for (s = 0; s < dbuf.nsample; s++) {
	data = l * s;
	/*	printf("[%d, %d] = %d, %d\n", l, s, data, dbuf.data.u16[l][s]); */
	if (dbuf.data.u16[l][s] != data){
	  good = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  nline = 10;
  nsample = 5;
  datatype = MTKe_float;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  for (l = 0; l < nline; l++) {
    lp = l * nsample;
    for (s = 0; s < nsample; s++) {
      data2 = 1.0 / ((l+1) * (s+1));
      ((float *)dataptr)[lp + s] = data2;
    }
  }

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_SUCCESS) {
    good = MTK_TRUE;
    for (l = 0; l < dbuf.nline; l++) {
      for (s = 0; s < dbuf.nsample; s++) {
	data2 = 1.0 / ((l+1) * (s+1));
	/*	printf("[%d, %d] = %f, %f\n", l, s, data2, dbuf.data.f[l][s]);*/
	if (fabsf(dbuf.data.f[l][s] - data2) > .000001){
	  good = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  nline = 13;
  nsample = 12;
  datatype = MTKe_double;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  for (l = 0; l < nline; l++) {
    lp = l * nsample;
    for (s = 0; s < nsample; s++) {
      data3 = 1.0 / ((l+1) * (s+1));
      ((double *)dataptr)[lp + s] = data3;
    }
  }

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_SUCCESS) {
    good = MTK_TRUE;
    for (l = 0; l < dbuf.nline; l++) {
      for (s = 0; s < dbuf.nsample; s++) {
	data3 = 1.0 / ((l+1) * (s+1));
	/*	printf("[%d, %d] = %f, %f\n", l, s, data3, dbuf.data.d[l][s]); */
	if (fabs(dbuf.data.d[l][s] - data3) > .000001 ){
	  good = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  nline = 5;
  nsample = 3;
  datatype = MTKe_int8;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  for (l = 0; l < nline; l++) {
    lp = l * nsample;
    for (s = 0; s < nsample; s++) {
      data4 = l * -s;
      ((char *)dataptr)[lp + s] = data4;
    }
  }

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_SUCCESS) {
    good = MTK_TRUE;
    for (l = 0; l < dbuf.nline; l++) {
      for (s = 0; s < dbuf.nsample; s++) {
	data4 = l * -s;
	/*	printf("[%d, %d] = %d, %d\n", l, s, data4, dbuf.data.i8[l][s]); */
	if (dbuf.data.i8[l][s] != data4){
	  good = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  nline = 1;
  nsample = 1;
  datatype = MTKe_float;
  dataptr = calloc(nline * nsample, datasize[datatype]);

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_SUCCESS) {
    good = MTK_TRUE;
    for (l = 0; l < dbuf.nline; l++) {
      for (s = 0; s < dbuf.nsample; s++) {
	/*	printf("[%d, %d] = %d, %d\n", l, s, 0.0, dbuf.data.i8[l][s]); */
	if (dbuf.data.f[l][s] != 0.0){
	  good = MTK_FALSE;
	}
      }
    }
    MtkDataBufferFree(&dbuf);
    free(dataptr);
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
  nline = -5;
  nsample = 3;
  datatype = MTKe_int8;

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  nline = 5;
  nsample = -3;
  datatype = MTKe_int8;

  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, &dbuf);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDataBufferImport(nline, nsample, datatype, dataptr, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkDataBufferImport(nline, nsample, datatype, NULL, &dbuf);
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
