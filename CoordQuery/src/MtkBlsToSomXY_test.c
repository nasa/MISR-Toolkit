/*===========================================================================
=                                                                           =
=                           MtkBlsToSomXY_test                              =
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
  int block;			/* Block */
  float line;			/* Fractional line */
  float sample;			/* Fractional sample */
  double somx;			/* SOM X */
  double somy;			/* SOM Y */
  double somx_expected;		/* Expected SOM X */
  double somy_expected;		/* Expected SOM Y */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkBlsToSomXY");

  /* Normal call test case */
  path = 1;
  res = 1100;
  block = 1;
  line = -0.5;
  sample = -0.5;
  somx_expected = 7460750.0;
  somy_expected = 527450.0;

  status = MtkBlsToSomXY(path, res, block, line, sample, &somx, &somy);
  /*
  printf("\n%d %20.12f %20.12f\n", block, line, sample);
  printf("%20.12f %20.12f\n", somx, somy);
  printf("%20.12f %20.12f\n", somx_expected, somy_expected);
  */
  if (status == MTK_SUCCESS &&
      fabs(somx - somx_expected) < 0.00001 &&
      fabs(somy - somy_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  path = 189;
  res = 275;
  block = 1;
  line = -0.5;
  sample = -0.5;
  somx_expected = 7460750.0;
  somy_expected = 527450.0;

  status = MtkBlsToSomXY(path, res, block, line, sample, &somx, &somy);
  /*
  printf("\n%d %20.12f %20.12f\n", block, line, sample);
  printf("%20.12f %20.12f\n", somx, somy);
  printf("%20.12f %20.12f\n", somx_expected, somy_expected);
  */
  if (status == MTK_SUCCESS &&
      fabs(somx - somx_expected) < 0.00001 &&
      fabs(somy - somy_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkBlsToSomXY(path, res, block, line, sample, NULL, &somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToSomXY(path, res, block, line, sample, &somx, NULL);
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
