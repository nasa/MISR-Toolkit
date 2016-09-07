/*===========================================================================
=                                                                           =
=                      MtkSetRegionByLatLonExtent_test                      =
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
  double ctr_lat_dd;		/* Center latitude in decimal degrees */
  double ctr_lon_dd;		/* Center longitude in decimal degrees */
  double lat_extent;		/* Latitude extent */
  double lon_extent;		/* Longitude extent */
  double meters_per_deg = 40007821.0 / 360.0;
				/* Average earth circumference */
				/* (40075004m + 39940638m) / 2.0 = 40007821m */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSetRegionByLatLonExtent");

  /* Normal test call */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100000.0;
  lon_extent = 1100000.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent/2)) < .0001){
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = -67.5;
  ctr_lon_dd = 25.0;
  lat_extent = 1650.0;
  lon_extent = 1100.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "kilometers",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*1000/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*1000/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 180.0;
  lat_extent = 20.0;
  lon_extent = 20.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "m",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 0.180;
  lon_extent = 0.360;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "km",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*1000/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*1000/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 180.0;
  lon_extent = 360.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "degrees",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*meters_per_deg/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*meters_per_deg/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 180.0;
  lon_extent = 360.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "deg",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*meters_per_deg/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*meters_per_deg/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 180.0;
  lon_extent = 360.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "dd",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*meters_per_deg/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*meters_per_deg/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 10.0;
  lon_extent = 200.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "275m",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*275/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*275/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 10.0;
  lon_extent = 200.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "275 meters",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*275/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*275/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 10.0;
  lon_extent = 200.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "1.1km",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*1100/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*1100/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  ctr_lat_dd = 0.0;
  ctr_lon_dd = 0.0;
  lat_extent = 10.0;
  lon_extent = 200.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "17.6 kilometers",
				      &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - ctr_lat_dd)) < .0001 &&
      (fabs(region.geo.ctr.lon - ctr_lon_dd)) < .0001 &&
      (fabs(region.hextent.xlat - lat_extent*17600/2)) < .0001 &&
      (fabs(region.hextent.ylon - lon_extent*17600/2)) < .0001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ctr_lat_dd = 95.0;
  ctr_lon_dd = 20.0;
  lat_extent = 10.0;
  lon_extent= 5.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ctr_lat_dd = -75.0;
  ctr_lon_dd = 20.0;
  lat_extent = -20.0;
  lon_extent = 30.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ctr_lat_dd = 75.0;
  ctr_lon_dd = 181.0;
  lat_extent = 60.0;
  lon_extent = 30.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ctr_lat_dd = 75.0;
  ctr_lon_dd = 20.0;
  lat_extent= 60.0;
  lon_extent = -20.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  ctr_lat_dd = -85.0;
  ctr_lon_dd = 20.0;
  lat_extent = 60.0;
  lon_extent = -20.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100000.0;
  lon_extent = 1100000.0;
  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "blah",
				      &region);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      NULL,
				      &region);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
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
