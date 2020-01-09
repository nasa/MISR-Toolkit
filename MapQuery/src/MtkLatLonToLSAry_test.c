/*===========================================================================
=                                                                           =
=                          MtkLatLonToLSAry_test                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrMapQuery.h"
#include "MisrSetRegion.h"
#include "MisrError.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Region region;		/* Region structure */
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT; /* Map Info structure */
  int path;			/* Path */
  int resolution;		/* Resolution */
  int sblock;			/* Start block */
  int eblock;			/* End block */
  double lat[3];		/* Latitude */
  double lon[3];		/* Longitude */
  int nelement = 3;		/* Number of elements */
  float line[3];		/* Line */
  float sample[3];		/* Sample */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToLSAry");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      lat[0] = 44.322089;
      lon[0] = -112.00821;
      lat[1] = 44.0;
      lon[1] = -112.0;
      lat[2] = 45.0;
      lon[2] = -113.0;

      /* Normal test call */
      status = MtkLatLonToLSAry(mapinfo, nelement, lat, lon, line, sample);
      if (status == MTK_SUCCESS &&
	  (line[0] - 704.0) < .0001 && (sample[0] - 312.0) < .0001 &&
	  (line[1] - 736.0554) < .0001 && (sample[1] - 317.6007) < .0001 &&
	  (line[2] - 704.0) < .0001 && (sample[2] - 312.0) < .0001) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      lat[1] = -44.0;

      /* Normal test call */
      status = MtkLatLonToLSAry(mapinfo, nelement, lat, lon, line, sample);
      if (status == MTK_OUTBOUNDS &&
	  (line[0] - 704.0) < .0001 && (sample[0] - 312.0) < .0001 &&
	  line[1] == -1.0 && sample[1] == -1.0 &&
	  (line[2] - 704.0) < .0001 && (sample[2] - 312.0) < .0001) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

    } else {
      pass = MTK_FALSE;
    }
  } else {
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkLatLonToLSAry(mapinfo, -1, lat, lon, line, sample);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToLSAry(mapinfo, nelement, NULL, lon, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToLSAry(mapinfo, nelement, lat, NULL, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToLSAry(mapinfo, nelement, lat, lon, NULL, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToLSAry(mapinfo, nelement, lat, lon, line, NULL);
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
