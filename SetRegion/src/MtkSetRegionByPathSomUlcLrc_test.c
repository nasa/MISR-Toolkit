/*===========================================================================
=                                                                           =
=                     MtkSetRegionByPathSomUlcLrc_test                      =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrSetRegion.h"
#include "MisrError.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Region region;		/* Region structure */
  double ulc_som_x;		/* Upper left latitude in decimal degrees */
  double ulc_som_y;		/* Upper left longitude in decimal degrees */
  double lrc_som_x;		/* Lower right latitude in decimal degrees */
  double lrc_som_y;		/* Lower right longitude in decimal degrees */
  int path;			/* Path */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSetRegionByPathSomUlcLrc");

  /* Normal test call */
  path = 27;
  ulc_som_x = 15600000.0;
  ulc_som_y = -300.0;
  lrc_som_x = 16800000.0;
  lrc_som_y = 2000.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 35.384617)) < .00001 &&
      (fabs(region.geo.ctr.lon - (-102.051381))) < .00001 &&
      (fabs(region.hextent.xlat - 600137.5)) < .00001 &&
      (fabs(region.hextent.ylon - 1287.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  path = 27;
  ulc_som_x = 20010000.0;
  ulc_som_y = -1000.0;
  lrc_som_x = 20450000.0;
  lrc_som_y = 300.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - (-0.885419))) < .00001 &&
      (fabs(region.geo.ctr.lon - (-104.841006))) < .00001 &&
      (fabs(region.hextent.xlat - 220137.5)) < .00001 &&
      (fabs(region.hextent.ylon - 787.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  path = 150;
  ulc_som_x = 20010000.0;
  ulc_som_y = -1000.0;
  lrc_som_x = 20450000.0;
  lrc_som_y = 300.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - -0.885419)) < .00001 &&
      (fabs(region.geo.ctr.lon - 65.116075)) < .00001 &&
      (fabs(region.hextent.xlat - 220137.5)) < .00001 &&
      (fabs(region.hextent.ylon - 787.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  path = 150;
  ulc_som_x = 20460000.0;
  ulc_som_y = -5000.0;
  lrc_som_x = 40000000.0;
  lrc_som_y = -1000.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - (-89.469396))) < .00001 &&
      (fabs(region.geo.ctr.lon - (-63-.694615))) < .00001 &&
      (fabs(region.hextent.xlat - 9770137.5)) < .00001 &&
      (fabs(region.hextent.ylon - 2137.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = 150;
  lrc_som_x = 20450000.0;
  ulc_som_y = -1000.0;
  ulc_som_y = 20010000.0;
  lrc_som_y = 300.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = 150;
  ulc_som_x = 20460000.0;
  ulc_som_y = -1000.0;
  lrc_som_x = 40000000.0;
  lrc_som_y = -5000.0;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = 421;
  ulc_som_x = 0.0;
  ulc_som_y = 0.0;
  lrc_som_x = 0.0;
  lrc_som_y = 0.;

  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkSetRegionByPathSomUlcLrc(path, ulc_som_x, ulc_som_y,
				       lrc_som_x, lrc_som_y,
				       NULL);
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
