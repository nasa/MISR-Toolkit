/*===========================================================================
=                                                                           =
=                           MtkCreateGeoGrid_test                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  int cn = 0;			/* Column number */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
				/* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
				/* Longitude data buffer structure */
  double ulc_lat_dd;		/* Upper left corner latitude in decimal deg */
  double ulc_lon_dd;		/* Upper left corner longitude in decimal deg */
  double lrc_lat_dd;		/* Lower right corner latitude in decimal deg */
  double lrc_lon_dd;		/* Lower right corner longitude in decimal deg*/
  double lat_cellsize_dd;	/* Latitude cellsize in decimal deg */
  double lon_cellsize_dd;	/* Longitude cellsize in decimal deg */
  int l;			/* Line index */
  int s;			/* Sample index */
  double lat_expected;		/* Expected latitude */
  double lon_expected;		/* Expected longitude */

  MTK_PRINT_STATUS(cn,"Testing MtkCreateGeoGrid");

  /* Normal test call */
  ulc_lat_dd = 30.0;
  ulc_lon_dd = -20.0;
  lrc_lat_dd = -30.0;
  lrc_lon_dd = 20.0;
  lat_cellsize_dd = 0.25;
  lon_cellsize_dd = 0.25;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_SUCCESS) {
    pass = MTK_TRUE;
    for (l=0; l < lat.nline; l++) {
      for (s=0; s < lat.nsample; s++) {
	lat_expected = ulc_lat_dd - l * lat_cellsize_dd;
	lon_expected = ulc_lon_dd + s * lon_cellsize_dd;
	if (fabs(lat.data.d[l][s] - lat_expected) > .00000001 ||
	    fabs(lon.data.d[l][s] - lon_expected) > .00000001) {
	  pass = MTK_FALSE;
	}
      }
    }
    if (pass) MTK_PRINT_STATUS(cn,".")
    else MTK_PRINT_STATUS(cn,"*");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ulc_lat_dd = 60.0;
  ulc_lon_dd = -160.0;
  lrc_lat_dd = 40.0;
  lrc_lon_dd = 160.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_SUCCESS) {
    pass = MTK_TRUE;
    for (l=0; l < lat.nline; l++) {
      for (s=0; s < lat.nsample; s++) {
	lat_expected = ulc_lat_dd - l * lat_cellsize_dd;
	lon_expected = ulc_lon_dd + s * lon_cellsize_dd;
	lon_expected = lon_expected > 180.0 ? lon_expected - 360.0 : lon_expected;
	if (fabs(lat.data.d[l][s] - lat_expected) > .00000001 ||
	    fabs(lon.data.d[l][s] - lon_expected) > .00000001) {
	  pass = MTK_FALSE;
	}
      }
    }
    if (pass) MTK_PRINT_STATUS(cn,".")
    else MTK_PRINT_STATUS(cn,"*");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 100.0;
  ulc_lon_dd = -160.0;
  lrc_lat_dd = 40.0;
  lrc_lon_dd = 160.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 60.0;
  ulc_lon_dd = -160.0;
  lrc_lat_dd = -100.0;
  lrc_lon_dd = 160.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = -60.0;
  ulc_lon_dd = -160.0;
  lrc_lat_dd = 40.0;
  lrc_lon_dd = 160.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 60.0;
  ulc_lon_dd = -190.0;
  lrc_lat_dd = 40.0;
  lrc_lon_dd = 160.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 60.0;
  ulc_lon_dd = -160.0;
  lrc_lat_dd = 40.0;
  lrc_lon_dd = 190.0;
  lat_cellsize_dd = 1.0;
  lon_cellsize_dd = 1.0;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 30.0;
  ulc_lon_dd = -20.0;
  lrc_lat_dd = -30.0;
  lrc_lon_dd = 20.0;
  lat_cellsize_dd = -0.25;
  lon_cellsize_dd = 0.25;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ulc_lat_dd = 30.0;
  ulc_lon_dd = -20.0;
  lrc_lat_dd = -30.0;
  lrc_lon_dd = 20.0;
  lat_cellsize_dd = 0.25;
  lon_cellsize_dd = -0.25;

  status = MtkCreateGeoGrid(ulc_lat_dd, ulc_lon_dd, lrc_lat_dd, lrc_lon_dd,
			    lat_cellsize_dd, lon_cellsize_dd, &lat, &lon);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
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
