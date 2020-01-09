/*===========================================================================
=                                                                           =
=                          MtkSomXYToLSAry_test                             =
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
  double som_x[3];		/* Som X */
  double som_y[3];		/* Som Y */
  int nelement = 3;		/* Number of elements */
  float line[3];		/* Line */
  float sample[3];		/* Sample */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSomXYToLSAry");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      som_x[0] = mapinfo.som.ulc.x;
      som_y[0] = mapinfo.som.ulc.y;
      som_x[1] = mapinfo.som.ctr.x;
      som_y[1] = mapinfo.som.ctr.y;
      som_x[2] = mapinfo.som.lrc.x;
      som_y[2] = mapinfo.som.lrc.y;

      /* Normal test call */
      status = MtkSomXYToLSAry(mapinfo, nelement, som_x, som_y,
			       line, sample);
      if (status == MTK_SUCCESS &&
	  line[0] == 0.0 && sample[0] == 0.0 &&
	  line[1] == 703.5 && sample[1] == 311.5 &&
	  line[2] == 1407.0 && sample[2] == 623.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      som_x[0] = mapinfo.som.lrc.x+2000;
      som_y[0] = mapinfo.som.lrc.y;

      /* Failure test call */
      status = MtkSomXYToLSAry(mapinfo, nelement, som_x, som_y, line, sample);
      if (status == MTK_OUTBOUNDS &&
	  line[0] == -1.0 && sample[0] == -1.0 &&
	  line[1] == 703.5 && sample[1] == 311.5 &&
	  line[2] == 1407.0 && sample[2] == 623.0) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      som_x[2] = mapinfo.som.lrc.x;
      som_y[2] = mapinfo.som.lrc.y+2000;

      /* Failure test call */
      status = MtkSomXYToLSAry(mapinfo, nelement, som_x, som_y, line, sample);
      if (status == MTK_OUTBOUNDS &&
	  line[0] == -1.0 && sample[0] == -1.0 &&
	  line[1] == 703.5 && sample[1] == 311.5 &&
	  line[2] == -1.0 && sample[2] == -1.0) {
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
  status = MtkSomXYToLSAry(mapinfo, -1, som_x, som_y, line, sample);
  if (status == MTK_BAD_ARGUMENT) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLSAry(mapinfo, nelement, NULL, som_y, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLSAry(mapinfo, nelement, som_x, NULL, line, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLSAry(mapinfo, nelement, som_x, som_y, NULL, sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkSomXYToLSAry(mapinfo, nelement, som_x, som_y, line, NULL);
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
