/*===========================================================================
=                                                                           =
=                            MtkLSToSomXY_test                              =
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
  double som_x;			/* Som X */
  double som_y;			/* Som Y */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLSToSomXY");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      /* Normal test call */
      status = MtkLSToSomXY(mapinfo, -0.5, -0.5, &som_x, &som_y);
      if (status == MTK_SUCCESS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLSToSomXY(mapinfo, 0.0, 0.0, &som_x, &som_y);
      if (status == MTK_SUCCESS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLSToSomXY(mapinfo, (mapinfo.nline-1)/2.0,
			    (mapinfo.nsample-1)/2.0, &som_x, &som_y);
      if (status == MTK_SUCCESS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLSToSomXY(mapinfo, mapinfo.nline-1, mapinfo.nsample-1,
			    &som_x, &som_y);
      if (status == MTK_SUCCESS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLSToSomXY(mapinfo, mapinfo.nline-0.5, mapinfo.nsample-0.5,
			    &som_x, &som_y);
      if (status == MTK_SUCCESS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Failure test call */
      status = MtkLSToSomXY(mapinfo, -1.0, 10.0, &som_x, &som_y);
      if (status == MTK_OUTBOUNDS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Failure test call */
      status = MtkLSToSomXY(mapinfo, 10, -1,  &som_x, &som_y);
      if (status == MTK_OUTBOUNDS) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkLSToSomXY(mapinfo, 10, -1,  NULL, &som_y);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLSToSomXY(mapinfo, 10, -1,  &som_x, NULL);
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
