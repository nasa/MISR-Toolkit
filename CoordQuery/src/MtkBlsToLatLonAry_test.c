/*===========================================================================
=                                                                           =
=                          MtkBlsToLatLonAry_test                           =
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
  int nelement;                 /* Number of elements in array */
  int block[2];			/* Block */
  float line[2];		/* Fractional line */
  float sample[2];		/* Fractional sample */
  double lat[2];		/* Latitude */
  double lon[2];		/* Longitude */
  double lat_expected[2];	/* Expected Latitude */
  double lon_expected[2];	/* Expected Longitude */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkBlsToLatLonAry");

  /* Normal call test case */
  path = 1;
  res = 1100;
  nelement = 1;
  block[0] = 1;
  line[0] = -0.5;
  sample[0] = -0.5;
  lat_expected[0] =  66.222667343954;
  lon_expected[0] = 110.46694104;

  status = MtkBlsToLatLonAry(path, res, nelement, block, line, sample,
			     lat, lon);
  /*
  printf("\n%d %20.12f %20.12f\n", block[0], line[0], sample[0]);
  printf("%20.12f %20.12f\n", lat[0], lon[0]);
  printf("%20.12f %20.12f\n", lat_expected[0], lon_expected[0]);
  */
  if (status == MTK_SUCCESS &&
      fabs(lat[0] - lat_expected[0]) < 0.00001 &&
      fabs(lon[0] - lon_expected[0]) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkBlsToLatLonAry(path, res, -1, block, line, sample,
			     lat, lon);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLonAry(path, res, nelement, NULL, line, sample,
			     lat, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLonAry(path, res, nelement, block, NULL, sample,
			     lat, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLonAry(path, res, nelement, block, line, NULL,
			     lat, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLonAry(path, res, nelement, block, line, sample,
			     NULL, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkBlsToLatLonAry(path, res, nelement, block, line, sample,
			     lat, NULL);
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
