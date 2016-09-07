/*===========================================================================
=                                                                           =
=                         MtkLatLonToSomXYAry_test                          =
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
  double lat[2];		/* Latitude array */
  double lon[2];		/* Longitude array */
  double somx[2];		/* SOM X array */
  double somy[2];		/* SOM Y array */
  double somx_expected[2];	/* Expected SOM X array */
  double somy_expected[2];	/* Expected SOM Y array */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToSomXYAry");

  /* Normal call test case */
  path = 1;
  nelement = 1;
  lat[0] = 66.226320603703;
  lon[0] = 110.452237414150;
  somx_expected[0] = 7461299.999957129359;
  somy_expected[0] = 527999.999819861725;

  status = MtkLatLonToSomXYAry(path, nelement, lat, lon, somx, somy);
  /*
  printf("\n%20.12f %20.12f %20.12f\n",lat[0], somx[0], somx_expected[0]);
  printf("\n%20.12f %20.12f %20.12f\n",lon[0], somy[0], somy_expected[0]);
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
  status = MtkLatLonToSomXYAry(path, -1, lat, lon, somx, somy);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToSomXYAry(path, nelement, NULL, lon, somx, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToSomXYAry(path, nelement, lat, NULL, somx, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToSomXYAry(path, nelement, lat, lon, NULL, somy);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToSomXYAry(path, nelement, lat, lon, somx, NULL);
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
