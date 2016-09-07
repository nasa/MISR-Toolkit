/*===========================================================================
=                                                                           =
=                          MtkLatLonToSomXY_test                            =
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
#include <float.h>
#include <stdio.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  int path;			/* Path */
  double lat;			/* Latitude */
  double lon;			/* Longitude */
  double somx;			/* SOM X */
  double somy;			/* SOM Y */
  double somx_expected;		/* Expected SOM X */
  double somy_expected;		/* Expected SOM Y */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToSomXY");

  /* Normal call test case */
  path = 1;
  lat = 66.226320603703;
  lon = 110.452237414150;
  somx_expected = 7461299.999957129359;
  somy_expected = 527999.999819861725;

  status = MtkLatLonToSomXY(path, lat, lon, &somx, &somy);
  /*
  printf("\n%20.12f %20.12f %20.12f\n",lat, somx, somx_expected);
  printf("\n%20.12f %20.12f %20.12f\n",lon, somy, somy_expected);
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
  lat = 65.821183366596;
  lon = 173.816760521356;
  somx_expected = 7495949.999777129851;
  somy_expected = 809049.999353842111;

  status = MtkLatLonToSomXY(path, lat, lon, &somx, &somy);
  /*
  printf("\n%20.12f %20.12f %20.12f\n",lat, somx, somx_expected);
  printf("\n%20.12f %20.12f %20.12f\n",lon, somy, somy_expected);
  */
  if (status == MTK_SUCCESS &&
      fabs(somx - somx_expected) < 0.00001 &&
      fabs(somy - somy_expected) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case with lat/lon out of bounds of block range,
     but not Som XY range.  This is still a successful call though. */
  path = 189;
  lat =  32.0;
  lon = 90.0;
  somx_expected = 11640816.350309696048;
  somy_expected = 7067250.059730839916;

  status = MtkLatLonToSomXY(path, lat, lon, &somx, &somy);
  /*
  printf("\n%20.12f %20.12f %20.12f %20.12f\n",lat, somx, somx_expected, fabs(somx-somx_expected));
  printf("\n%20.12f %20.12f %20.12f %20.12f\n",lon, somy, somy_expected, fabs(somy-somy_expected));
  */
  if (status == MTK_SUCCESS &&
      fabs(somx - somx_expected) < 0.0001 &&
      fabs(somy - somy_expected) < 0.0002) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */

  status = MtkLatLonToSomXY(path, lat, lon, NULL, &somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToSomXY(path, lat, lon, &somx, NULL);
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
