/*===========================================================================
=                                                                           =
=                      MtkRegionPathToBlockRange_test                       =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrError.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double lat;			/* Latitude */
  double lon;			/* Longitude */
  MTKt_Region region;		/* Region */
  int sb;			/* Start block */
  int eb;			/* End block */
  int sbp;			/* Start block */
  int ebp;			/* End block */
  int path;			/* Path */
  int pathcnt;			/* Path count */
  int *pathlist;		/* Path list */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRegionPathToBlockRange");

  /* Normal call test case */
  lat = 33.2;
  lon = -112.5;

  MtkSetRegionByLatLonExtent(lat, lon, 1000000, 500000, "meters", &region);
  path = 37;

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 60 && eb == 68) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  66.222667343954;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 1 && eb == 3) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  0.0;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 89 && eb == 92) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  40.5;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 57 && eb == 59) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  0.0;
  lon =  0.0;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 89 && eb == 92) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  0.0;
  lon =  -180.0;

  MtkSetRegionByLatLonExtent(lat, lon, 3600000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 78 && eb == 103) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  0.0;
  lon =  179.0;

  MtkSetRegionByLatLonExtent(lat, lon, 3600000, 500000, "meters", &region);
  MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  path = pathlist[pathcnt/2];
  free(pathlist);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 78 && eb == 103) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  path = 38;
  sbp = 50;
  ebp = 60;

  MtkSetRegionByPathBlockRange(path, sbp, ebp, &region);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 50 && eb == 60) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal call test case */
  lat =  10.0;
  lon =  20.0;
  path = 181;

  MtkSetRegionByLatLonExtent(lat, lon, 3000, 4000, "meters", &region);

  status = MtkRegionPathToBlockRange(region, path, &sb, &eb);
  if (status == MTK_SUCCESS &&
      sb == 82 && eb == 83) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkRegionPathToBlockRange(region, path, NULL, &eb);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkRegionPathToBlockRange(region, path, &sb, NULL);
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

