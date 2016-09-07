/*===========================================================================
=                                                                           =
=                        MtkSetRegionByUlcLrc_test                          =
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
  double ulc_lat_dd;		/* Upper left latitude in decimal degrees */
  double ulc_lon_dd;		/* Upper left longitude in decimal degrees */
  double lrc_lat_dd;		/* Lower right latitude in decimal degrees */
  double lrc_lon_dd;		/* Lower right longitude in decimal degrees */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSetRegionByUlcLrc");

  /* Normal test call */
  ulc_lat_dd = 40.0;
  ulc_lon_dd = -120.0;
  lrc_lat_dd = 30.0;
  lrc_lon_dd = -110.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 35.0)) < .00001 &&
      (fabs(region.geo.ctr.lon - (-115.0))) < .00001 &&
      (fabs(region.hextent.xlat - 556597.715754)) < .00001 &&
      (fabs(region.hextent.ylon - 556597.715754)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ulc_lat_dd = -60.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = -75.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - (-67.5))) < .00001 &&
      (fabs(region.geo.ctr.lon - 25.0)) < .00001 &&
      (fabs(region.hextent.xlat - 834896.57362)) < .00001 &&
      (fabs(region.hextent.ylon - 556597.71575)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ulc_lat_dd = 10.0;
  ulc_lon_dd = 170.0;
  lrc_lat_dd = -10.0;
  lrc_lon_dd = -170.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 0.0)) < .00001 &&
      (fabs(region.geo.ctr.lon - 180.0)) < .00001 &&
      (fabs(region.hextent.xlat - 1113195.43149)) < .00001 &&
      (fabs(region.hextent.ylon - 1113195.43149)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ulc_lat_dd = 90.0;
  ulc_lon_dd = -180.0;
  lrc_lat_dd = -90.0;
  lrc_lon_dd = 180.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 0.0)) < .00001 &&
      (fabs(region.geo.ctr.lon - 0.0)) < .00001 &&
      (fabs(region.hextent.xlat - 10018758.8835)) < .00001 &&
      (fabs(region.hextent.ylon - 20037517.767)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 95.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = -60.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = -75.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = -99.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 75.0;
  ulc_lon_dd = 181.0;
  lrc_lat_dd = 60.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 75.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = 60.0;
  lrc_lon_dd = -180.1;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = -75.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = -60.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = -75.0;
  ulc_lon_dd = 20.0;
  lrc_lat_dd = -75.0;
  lrc_lon_dd = 30.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = -75.0;
  ulc_lon_dd = -20.0;
  lrc_lat_dd = -60.0;
  lrc_lon_dd = -20.0;

  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
				&region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				lrc_lat_dd, lrc_lon_dd,
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
