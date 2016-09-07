/*===========================================================================
=                                                                           =
=                          MtkBlsToSomXYAry_test                            =
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

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int path;			/* Path */
  int res;			/* Resolution in meters */
  int nelement;			/* Number of elements in array */
  int block[2];			/* Block */
  float line[2];		/* Fractional line */
  float sample[2];		/* Fractional sample */
  double somx[2];		/* SOM X */
  double somy[2];		/* SOM Y */
  double somx_expected[2];	/* Expected SOM X */
  double somy_expected[2];	/* Expected SOM Y */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkBlsToSomXYAry");

  /* Normal call test case */
  path = 1;
  res = 1100;
  nelement = 1;
  block[0] = 1;
  line[0] = -0.5;
  sample[0] = -0.5;
  somx_expected[0] = 7460750.0;
  somy_expected[0] = 527450.0;

  status = MtkBlsToSomXYAry(path, res, nelement, block, line, sample, 
			    somx, somy);
  /*
  printf("\n%d %20.12f %20.12f\n", block[0], line[0], sample[0]);
  printf("%20.12f %20.12f\n", somx[0], somy[0]);
  printf("%20.12f %20.12f\n", somx_expected[0], somy_expected[0]);
  */
  if (status == MTK_SUCCESS &&
      fabs(somx[0] - somx_expected[0]) < 0.00001 &&
      fabs(somy[0] - somy_expected[0]) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkBlsToSomXYAry(path, res, -1, block, line, sample, 
			    somx, somy);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXYAry(path, res, nelement, NULL, line, sample, 
			    somx, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXYAry(path, res, nelement, block, NULL, sample, 
			    somx, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXYAry(path, res, nelement, block, line, NULL, 
			    somx, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXYAry(path, res, nelement, block, line, sample, 
			    NULL, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXYAry(path, res, nelement, block, line, sample, 
			    somx, NULL);
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
