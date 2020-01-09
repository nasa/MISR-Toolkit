/*===========================================================================
=                                                                           =
=                           MtkLatLonToBls_test                             =
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
  double lat;			/* Latitude */
  double lon;			/* Longitude */
  int block;                    /* Block */
  float line;                   /* Fractional line */
  float sample;                 /* Fractional sample */
  int block_expected;           /* Expected block */
  float line_expected;          /* Expected fractional line */
  float sample_expected;        /* Expected fractional sample */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToBls");

  /* Normal call test case */
  path = 1;
  res = 1100;
  lat = 66.226320603703;
  lon = 110.452237414150;
  block_expected = 1;
  line_expected = 0.0;
  sample_expected = 0.0;

  status = MtkLatLonToBls(path, res, lat, lon, &block, &line, &sample);
  /*
  printf("\n%20.12f %20.12f\n", lat, lon);
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
  lat = 65.821183366596;
  lon = 173.816760521356;
  block_expected = 1;
  line_expected = 127.5;
  sample_expected = 1023.5;

  status = MtkLatLonToBls(path, res, lat, lon, &block, &line, &sample);
  /*
  printf("\n%20.12f %20.12f\n", lat, lon);
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

  /* Argument Checks */
  status = MtkLatLonToBls(path, res, lat, lon, NULL, &line, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBls(path, res, lat, lon, &block, NULL, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBls(path, res, lat, lon, &block, &line, NULL);
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
