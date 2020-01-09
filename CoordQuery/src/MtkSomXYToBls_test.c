/*===========================================================================
=                                                                           =
=                            MtkSomXYToBls_test                             =
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
  double somx;			/* SOM X */
  double somy;			/* SOM Y */
  int block;			/* Block */
  float line;			/* Fractional line */
  float sample;			/* Fractional sample */
  int block_expected;		/* Expected block */
  float line_expected;		/* Expected fractional line */
  float sample_expected;	/* Expected fractional sample */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToBls");

  /* Normal call test case */
  path = 1;
  res = 1100;
  somx = 7460750.0;
  somy = 527450.0;
  block_expected = 1;
  line_expected = -0.5;
  sample_expected = -0.5;

  status = MtkSomXYToBls(path, res, somx, somy, &block, &line, &sample);
  /*
  printf("\n%20.12f %20.12f\n", somx, somy);
  printf("%d %20.12f %20.12f\n", block, line, sample);
  printf("%d %20.12f %20.12f\n", block_expected, line_expected, sample_expected);
  */
  if (status == MTK_SUCCESS &&
      abs(block - block_expected) == 0 &&
      fabs(line - line_expected) < 0.00001 &&
      fabs(sample - sample_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  path = 189;
  res = 275;
  somx = 7460750.0;
  somy = 527450.0;
  block_expected = 1;
  line_expected = -0.5;
  sample_expected = -0.5;

  status = MtkSomXYToBls(path, res, somx, somy, &block, &line, &sample);
  /*
  printf("\n%20.12f %20.12f\n", somx, somy);
  printf("%d %20.12f %20.12f\n", block, line, sample);
  printf("%d %20.12f %20.12f\n", block_expected, line_expected, sample_expected);
  */
  if (status == MTK_SUCCESS &&
      abs(block - block_expected) == 0 &&
      fabs(line - line_expected) < 0.00001 &&
      fabs(sample - sample_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case with Som XY out of bounds of block range */
  path = 189;
  res = 275;
  somx = 7460749.0;
  somy = 527449.0;
  block_expected = -1;
  line_expected = -1.0;
  sample_expected = -1.0;

  status = MtkSomXYToBls(path, res, somx, somy, &block, &line, &sample);
  /*
  printf("\n%20.12f %20.12f\n", somx, somy);
  printf("%d %20.12f %20.12f\n", block, line, sample);
  printf("%d %20.12f %20.12f\n", block_expected, line_expected, sample_expected);
  */
  if (status == MTK_MISR_FORWARD_PROJ_FAILED &&
      abs(block - block_expected) == 0 &&
      fabs(line - line_expected) < 0.00001 &&
      fabs(sample - sample_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkSomXYToBls(path, res, somx, somy, NULL, &line, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBls(path, res, somx, somy, &block, NULL, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToBls(path, res, somx, somy, &block, &line, NULL);
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
