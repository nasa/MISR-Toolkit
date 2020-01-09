/*===========================================================================
=                                                                           =
=                          MtkSomXYToBlsAry_test                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrError.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int path;			/* Path */
  int res;			/* Resolution in meters */
  int nelement;			/* Number of elements in array */
  double somx[2];		/* SOM X */
  double somy[2];		/* SOM Y */
  int block[2];			/* Block */
  float line[2];		/* Fractional line */
  float sample[2];		/* Fractional sample */
  int block_expected[2];	/* Expected block */
  float line_expected[2];	/* Expected fractional line */
  float sample_expected[2];	/* Expected fractional sample */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToBlsAry");

  /* Normal call test case */
  path = 1;
  res = 1100;
  nelement = 1;
  somx[0] = 7460750.0;
  somy[0] = 527450.0;
  block_expected[0] = 1;
  line_expected[0] = -0.5;
  sample_expected[0] = -0.5;

  status = MtkSomXYToBlsAry(path, res, nelement, somx, somy, 
			    block, line, sample);
  /*
  printf("\n%20.12f %20.12f\n", somx[0], somy[0]);
  printf("%d %20.12f %20.12f\n", block[0], line[0], sample[0]);
  printf("%d %20.12f %20.12f\n", block_expected[0], line_expected[0], sample_expected[0]);
  */
  if (status == MTK_SUCCESS &&
      abs(block[0] - block_expected[0]) == 0 &&
      fabs(line[0] - line_expected[0]) < 0.00001 &&
      fabs(sample[0] - sample_expected[0]) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkSomXYToBlsAry(path, res, -1, somx, somy, 
			    block, line, sample);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBlsAry(path, res, nelement, NULL, somy, 
			    block, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBlsAry(path, res, nelement, somx, NULL, 
			    block, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBlsAry(path, res, nelement, somx, somy, 
			    NULL, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBlsAry(path, res, nelement, somx, somy, 
			    block, NULL, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBlsAry(path, res, nelement, somx, somy, 
			    block, line, NULL);
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
