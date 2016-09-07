/*===========================================================================
=                                                                           =
=                     MtkSetRegionByPathBlockRange_test                     =
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
  int path;			/* Path */
  int start_block;		/* Start_block */
  int end_block;		/* End_block */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSetRegionByPathBlockRange");

  /* Normal test call */
  path = 39;
  start_block = 50;
  end_block = 60;

  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 44.32774)) < .00001 &&
      (fabs(region.geo.ctr.lon - (-112.01395))) < .00001 &&
      (fabs(region.hextent.xlat - 774262.5)) < .00001 &&
      (fabs(region.hextent.ylon - 343062.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  path = 233;
  start_block = 10;
  end_block = 170;

  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, &region);
  if (status == MTK_SUCCESS &&
      (fabs(region.geo.ctr.lat - 0.62052)) < .00001 &&
      (fabs(region.geo.ctr.lon - (-62.93331))) < .00001 &&
      (fabs(region.hextent.xlat - 11334262.5)) < .00001 &&
      (fabs(region.hextent.ylon - 1152662.5)) < .00001) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = -1;
  start_block = 10;
  end_block = 170;

  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = 65;
  start_block = 170;
  end_block = 10;

  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Failure test call */
  path = 65;
  start_block = 10;
  end_block = 181;

  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, &region);
  if (status == MTK_OUTBOUNDS) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Check */
  status = MtkSetRegionByPathBlockRange(path, start_block, end_block, NULL);
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
