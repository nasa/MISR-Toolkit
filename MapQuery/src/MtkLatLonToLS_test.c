/*===========================================================================
=                                                                           =
=                            MtkLatLonToLS_test                             =
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

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToLS");

  path = 39;
  resolution = 1100;
  sblock = 50;
  eblock = 60;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status == MTK_SUCCESS) {
    status = MtkSnapToGrid(path, resolution, region, &mapinfo);
    if (status == MTK_SUCCESS) {

      /* Normal test call */
      status = MtkLatLonToLS(mapinfo, 44.322089, -112.00821, &line, &sample);
      if (status == MTK_SUCCESS &&
	  (line - 704.0) < .0001 && 
	  (sample - 312.0) < .0001) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLatLonToLS(mapinfo, 44.0, -112.0, &line, &sample);
      if (status == MTK_SUCCESS &&
	  (line - 736.0554) < .0001 && 
	  (sample - 317.6007) < .0001) {
	MTK_PRINT_STATUS(cn,".");
      } else {
	MTK_PRINT_STATUS(cn,"*");
	pass = MTK_FALSE;
      }

      /* Normal test call */
      status = MtkLatLonToLS(mapinfo, -44.0, -112.0, &line, &sample);
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
  status = MtkLatLonToLS(mapinfo, 48.9, -100.3, NULL, &sample);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToLS(mapinfo, 48.9, -100.3, &line, NULL);
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
