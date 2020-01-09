/*===========================================================================
=                                                                           =
=                            MtkSomXYToLS_test                              =
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
  float line;			/* Line */
  float sample;			/* Sample */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToLS");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      /* Normal test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.ulc.x - resolution/2,
			    mapinfo.som.ulc.y - resolution/2, &line, &sample);
      if (status == MTK_SUCCESS &&
	  line == -0.5 &&
	  sample == -0.5) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.ulc.x,
			    mapinfo.som.ulc.y, &line, &sample);
      if (status == MTK_SUCCESS &&
	  line == 0.0 &&
	  sample == 0.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.ctr.x,
			    mapinfo.som.ctr.y, &line, &sample);
      if (status == MTK_SUCCESS &&
	  line == 703.5 &&
	  sample == 311.5) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x,
			    mapinfo.som.lrc.y, &line, &sample);
      if (status == MTK_SUCCESS &&
	  line == 1407.0 &&
	  sample == 623.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x + resolution/2,
			    mapinfo.som.lrc.y + resolution/2, &line, &sample);
      if (status == MTK_SUCCESS &&
	  line == 1407.5 &&
	  sample == 623.5) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Failure test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x+2000,
			    mapinfo.som.lrc.y, &line, &sample);
      if (status == MTK_OUTBOUNDS &&
	  line == -1.0 &&
	  sample == -1.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Failure test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x,
			    mapinfo.som.lrc.y+2000, &line, &sample);
      if (status == MTK_OUTBOUNDS &&
	  line == -1.0 &&
	  sample == -1.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Failure test call */
      status = MtkSomXYToLS(mapinfo, mapinfo.som.ulc.x-2000,
			    mapinfo.som.ulc.y, &line, &sample);
      if (status == MTK_OUTBOUNDS &&
	  line == -1.0 &&
	  sample == -1.0) {
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
  status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x,
   		        mapinfo.som.lrc.y+2000, NULL, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLS(mapinfo, mapinfo.som.lrc.x,
   		        mapinfo.som.lrc.y+2000, &line, NULL);
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
