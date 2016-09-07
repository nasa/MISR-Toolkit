/*===========================================================================
=                                                                           =
=                           MtkCreateLatLon_test                            =
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
#include "MisrUtil.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

int main () {

  MTKt_status status;           /* Return status */
  int cn = 0;			/* Column number */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_Region region = MTKT_REGION_INIT;
				/* Region structure */
  MTKt_MapInfo mapinfo = MTKT_MAPINFO_INIT;
				/* Map Info structure */
  MTKt_DataBuffer lat = MTKT_DATABUFFER_INIT;
				/* Latitude data buffer structure */
  MTKt_DataBuffer lon = MTKT_DATABUFFER_INIT;
				/* Longitude data buffer structure */

  MTK_PRINT_STATUS(cn,"Testing MtkCreateLatLon");

  MtkSetRegionByPathBlockRange(37, 35, 36, &region);
  MtkSnapToGrid(37, 1100, region, &mapinfo);

  /* Normal test call */
  status = MtkCreateLatLon(mapinfo, &lat, &lon);
  if (status == MTK_SUCCESS &&
      fabs(lat.data.d[63][79] - 69.025150217462524) < .00001 &&
      fabs(lon.data.d[63][79] - -98.336073888584707) < .00001) {
    MTK_PRINT_STATUS(cn,".");
    MtkDataBufferFree(&lat);
    MtkDataBufferFree(&lon);
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  MtkSetRegionByPathBlockRange(233, 1, 180, &region);
  MtkSnapToGrid(233, 17600, region, &mapinfo);

  /* Normal test call */
  status = MtkCreateLatLon(mapinfo, &lat, &lon);
  if (status == MTK_SUCCESS &&
      fabs(lat.data.d[45][97] - 73.000596583683134) < .00001 &&
      fabs(lon.data.d[45][97] - 105.09908276358931) < .00001) {
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
