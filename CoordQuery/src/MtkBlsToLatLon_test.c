/*===========================================================================
=                                                                           =
=                           MtkBlsToLatLon_test                             =
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
  double lat;                   /* Latitude */
  double lon;                   /* Longitude */
  double lat_expected;          /* Expected Latitude */
  double lon_expected;          /* Expected Longitude */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkBlsToLatLon");

  /* Normal call test case */
  path = 1;
  res = 1100;
  block = 1;
  line = 0.0;
  sample = 0.0;
  lat_expected = 66.226320603703;
  lon_expected = 110.452237414150;

  status = MtkBlsToLatLon(path, res, block, line, sample, &lat, &lon);
  /*
  printf("\n%d %20.12f %20.12f\n", block, line, sample);
  printf("%20.12f %20.12f\n", lat, lon);
  printf("%20.12f %20.12f\n", lat_expected, lon_expected);
  */
  if (status == MTK_SUCCESS &&
      fabs(lat - lat_expected) < 0.00001 &&
      fabs(lon - lon_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  path = 189;
  res = 275;
  block = 1;
  line = 127.5;
  sample = 1023.5;
  lat_expected = 65.821183366596;
  lon_expected = 173.816760521356;

  status = MtkBlsToLatLon(path, res, block, line, sample, &lat, &lon);
  /*
  printf("\n%d %20.12f %20.12f\n", block, line, sample);
  printf("%20.12f %20.12f\n", lat, lon);
  printf("%20.12f %20.12f\n", lat_expected, lon_expected);
  */
  if (status == MTK_SUCCESS &&
      fabs(lat - lat_expected) < 0.00001 &&
      fabs(lon - lon_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkBlsToLatLon(path, res, block, line, sample, NULL, &lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLon(path, res, block, line, sample, &lat, NULL);
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
