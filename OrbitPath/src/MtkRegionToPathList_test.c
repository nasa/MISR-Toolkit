/*===========================================================================
=                                                                           =
=                         MtkRegionToPathList_test                          =
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
#include "MisrProjParam.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int checkpathlist( int pathcnt, int *pathlist, int *pathlist_expected );

int checkpathsequence( int pathcnt, int *pathlist, int startpath, int endpath );

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double lat;                   /* Latitude */
  double lon;                   /* Longitude */
  double ulclat;                /* Latitude */
  double ulclon;                /* Longitude */
  double lrclat;                /* Latitude */
  double lrclon;                /* Longitude */
  MTKt_Region region;           /* Region */
  int pathcnt;                  /* Path count */
  int *pathlist;                /* Path list */

  /* Expected path list for each test case */
  int pathlist_expected_case1[] = { 1,2,130,131,132,133,134,135,\
                                    136,137,138,139,140,141,142,\
                                    224,225,226,227,228,229,230,\
                                    231,232,233 };
  int pathlist_expected_case2[] = { 118,119,120,121,122,123,124 };
  int pathlist_expected_case3[] = { 124,125,126,127,128,129,130,131 };
  int pathlist_expected_case4[] = { 189,190,191,192,193,194,195 };
  int pathlist_expected_case5[] = { 73,74,75,76,77,78 };
  int pathlist_expected_case6[] = { 74,75,76,77,78,79 };
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkRegionToPathList");

  /* Normal call test case */
  lat =  66.222667343954;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case1)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
  lat =  0.0;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case2)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
  lat =  40.5;
  lon = 110.46694104;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case3)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
  lat =  0.0;
  lon =  0.0;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case4)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
  lat =  0.0;
  lon =  -180.0;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case5)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
  lat =  0.0;
  lon =  179.0;

  MtkSetRegionByLatLonExtent(lat, lon, 360000, 500000, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case6)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Normal call test case */
    ulclat = 30.0;
    ulclon = -180.0;
    lrclat = 20.0;
    lrclon = 180.0;


    MtkSetRegionByUlcLrc(ulclat, ulclon, lrclat, lrclon, &region);

    status = MtkRegionToPathList(region, &pathcnt, &pathlist);
    if (status == MTK_SUCCESS &&
        checkpathsequence(pathcnt, pathlist, 1, 233)) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
    free(pathlist);

    /* Normal call test case */
    ulclat = 5.0;
    ulclon = -90.0;
    lrclat = -5.0;
    lrclon = 90.0;


    MtkSetRegionByUlcLrc(ulclat, ulclon, lrclat, lrclon, &region);

    status = MtkRegionToPathList(region, &pathcnt, &pathlist);
    if (status == MTK_SUCCESS &&
        checkpathsequence(pathcnt, pathlist, 132, 19)) {
      MTK_PRINT_STATUS(cn,".");
    } else {
      MTK_PRINT_STATUS(cn,"*");
      pass = MTK_FALSE;
    }
    free(pathlist);

  /* Failure call test case */
  lat =  87.0;
  lon =  20.0;

  MtkSetRegionByLatLonExtent(lat, lon, 360, 500, "meters", &region);

  status = MtkRegionToPathList(region, &pathcnt, &pathlist);
  if (status == MTK_NOT_FOUND) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkRegionToPathList(region, NULL, &pathlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkRegionToPathList(region, &pathcnt, NULL);
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

int checkpathlist( int pathcnt, int *pathlist, int *pathlist_expected ) {

  int i;			/* Loop index */

  for (i = 0; i < pathcnt; i++) {
    if (pathlist[i] != pathlist_expected[i]) return MTK_FALSE;
  }

  return MTK_TRUE;
}

int checkpathsequence( int pathcnt, int *pathlist, int startpath, int endpath ) {
  int i;
  if (endpath < startpath) {
    for (i = 0; i < endpath; i++) {
      if (pathlist[i] != i + 1 ) return MTK_FALSE;
    }
    for (; i < pathcnt; i++) {
      if (pathlist[i] != startpath++ ) return MTK_FALSE;
    }
  } else {
    if ((endpath - startpath + 1) != pathcnt) return MTK_FALSE;
    for (i = 0; i < pathcnt; i++) {
      if (pathlist[i] != startpath + i ) return MTK_FALSE;
    }
  }
  return MTK_TRUE;
}
