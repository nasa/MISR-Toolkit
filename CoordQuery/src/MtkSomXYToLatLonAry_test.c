/*===========================================================================
=                                                                           =
=                         MtkSomXYToLatLonAry_test                          =
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
  int nelement;			/* Number of elements in array */
  double somx[2];		/* SOM X array */
  double somy[2];		/* SOM Y array */
  double lat[2];		/* Latitude array */
  double lon[2];		/* Longitude array */
  double lat_expected[2];	/* Expected Latitude array */
  double lon_expected[2];	/* Expected Longitude array */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToLatLonAry");

  /* Normal call test case */
  path = 1;
  nelement = 1;
  somx[0] = 7460750.0;
  somy[0] = 527450.0;
  lat_expected[0] =  66.222667343954;
  lon_expected[0] = 110.46694104;

  status = MtkSomXYToLatLonAry(path, nelement, somx, somy, lat, lon);
  /*
  printf("\n%20.12f %20.12f %20.12f\n", lat[0], lat_expected[0], somx[0]);
  printf("\n%20.12f %20.12f %20.12f\n", lon[0], lon_expected[0], somy[0]);
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
  status = MtkSomXYToLatLonAry(path, -1, somx, somy, lat, lon);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLatLonAry(path, nelement, NULL, somy, lat, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLatLonAry(path, nelement, somx, NULL, lat, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLatLonAry(path, nelement, somx, somy, NULL, lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLatLonAry(path, nelement, somx, somy, lat, NULL);
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
