/*===========================================================================
=                                                                           =
=                          MtkLatLonToBlsAry_test                           =
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
  int nelement;                 /* Number of elements in array */
  double lat[2];		/* Latitude */
  double lon[2];		/* Longitude */
  int block[2];			/* Block */
  float line[2];		/* Fractional line */
  float sample[2];		/* Fractional sample */
  int block_expected[2];	/* Expected block */
  float line_expected[2];	/* Expected fractional line */
  float sample_expected[2];	/* Expected fractional sample */
  int cn = 0;                   /* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToBlsAry");

  /* Normal call test case */
  path = 1;
  res = 1100;
  nelement = 1;
  lat[0] = 66.226320603703;
  lon[0] = 110.452237414150;
  block_expected[0] = 1;
  line_expected[0] = 0.0;
  sample_expected[0] = 0.0;

  status = MtkLatLonToBlsAry(path, res, nelement, lat, lon,
			     block, line, sample);
  /*
  printf("\n%20.12f %20.12f\n", lat[0], lon[0]);
  printf("%d %20.12f %20.12f\n", block[0], line[0], sample[0]);
  printf("%d %20.12f %20.12f\n", block_expected[0], line_expected[0], sample_expected[0]);
  */
  if (status == MTK_SUCCESS &&
      abs(block[0] - block_expected[0]) == 0 &&
      fabs(line[0] - line_expected[0]) < 0.00001 &&
      fabs(sample[0] - sample_expected[0]) < 0.00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkLatLonToBlsAry(path, res, -1, lat, lon,
			     block, line, sample);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBlsAry(path, res, nelement, NULL, lon,
			     block, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBlsAry(path, res, nelement, lat, NULL,
			     block, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBlsAry(path, res, nelement, lat, lon,
			     NULL, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBlsAry(path, res, nelement, lat, lon,
			     block, NULL, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToBlsAry(path, res, nelement, lat, lon,
			     block, line, NULL);
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
