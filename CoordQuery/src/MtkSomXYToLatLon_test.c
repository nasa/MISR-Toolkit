/*===========================================================================
=                                                                           =
=                          MtkSomXYToLatLon_test                            =
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
  double somx;			/* SOM X */
  double somy;			/* SOM Y */
  double lat;			/* Latitude */
  double lon;			/* Longitude */
  double lat_expected;		/* Expected Latitude */
  double lon_expected;		/* Expected Longitude */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToLatLon");

  /* Normal call test case */
  path = 1;
  somx = 7461299.999957129359;
  somy = 527999.999819861725;
  lat_expected = 66.226320603703;
  lon_expected = 110.452237414150;

  status = MtkSomXYToLatLon(path, somx, somy, &lat, &lon);
  /*
  printf("\n%20.12f %20.12f %20.12f\n", lat, lat_expected, somx);
  printf("\n%20.12f %20.12f %20.12f\n", lon, lon_expected, somy);
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
  somx = 7495949.999777129851;
  somy = 809049.999353842111;
  lat_expected = 65.821183366596;
  lon_expected = 173.816760521356;

  status = MtkSomXYToLatLon(path, somx, somy, &lat, &lon);
  /*
  printf("\n%20.12f %20.12f %20.12f\n", lat, lat_expected, somx);
  printf("\n%20.12f %20.12f %20.12f\n", lon, lon_expected, somy);
  */
  if (status == MTK_SUCCESS &&
      fabs(lat - lat_expected) < 0.00001 &&
      fabs(lon - lon_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case with Som XY out of bounds of block range,
     but not lat/lon range.  This is still a successful call though. */
  path = 189;
  somx = 11640816.350309696048;
  somy = 7067250.059730839916;
  lat_expected = 32.0;
  lon_expected = 90.0;

  status = MtkSomXYToLatLon(path, somx, somy, &lat, &lon);
  /*
  printf("\n%20.12f %20.12f %20.12f %20.12f\n", lat, lat_expected, somx, fabs(lat-lat_expected));
  printf("\n%20.12f %20.12f %20.12f %20.12f\n", lon, lon_expected, somy, fabs(lon-lon_expected));
  */
  if (status == MTK_SUCCESS &&
      fabs(lat - lat_expected) < 0.001 &&
      fabs(lon - lon_expected) < 0.001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkSomXYToLatLon(path, somx, somy, NULL, &lon);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLatLon(path, somx, somy, &lat, NULL);
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
