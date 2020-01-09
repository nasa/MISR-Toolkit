/*===========================================================================
=                                                                           =
=                          MtkLSToLatLonAry_test                            =
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
  float line[3];		/* Line */
  float sample[3];		/* Sample */
  int nelement = 3;		/* Number of elements */
  double lat_dd[3];		/* Latitude */
  double lon_dd[3];		/* Longitude */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLSToLatLonAry");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      line[0] = 0.0;
      sample[0] = 0.0;
      line[1] = (mapinfo.nline-1)/2.0;
      sample[1] = (mapinfo.nsample-1)/2.0;
      line[2] = mapinfo.nline-1;
      sample[2] = mapinfo.nsample-1;

      /* Normal test call */
      status = MtkLSToLatLonAry(mapinfo, nelement, line, sample,
				lat_dd, lon_dd);
      if (status == MTK_SUCCESS) {
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
  status = MtkLSToLatLonAry(mapinfo, -1, line, sample,
			    lat_dd, lon_dd);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLSToLatLonAry(mapinfo, nelement, NULL, sample,
			    lat_dd, lon_dd);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLSToLatLonAry(mapinfo, nelement, line, NULL,
			    lat_dd, lon_dd);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLSToLatLonAry(mapinfo, nelement, line, sample,
			    NULL, lon_dd);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLSToLatLonAry(mapinfo, nelement, line, sample,
			    lat_dd, NULL);
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
