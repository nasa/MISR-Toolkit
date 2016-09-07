/*===========================================================================
=                                                                           =
=                         MtkLatLonToPathList_test                          =
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

int main () {

  MTKt_status status;		/* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  double lat;			/* Latitude */
  double lon;			/* Longitude */
  int pathcnt;			/* Path count */
  int *pathlist;		/* Path list */
				/* Expected path list for each test case */
  int pathlist_expected_case1[] = { 133,134,135,136,137,138,139,140, \
                                    226,227,228,229,230,231,232,233 };
  int pathlist_expected_case2[] = { 120,121,122 };
  int pathlist_expected_case3[] = { 125,126,127,128,129 };
  int pathlist_expected_case4[] = { 191,192,193 };
  int pathlist_expected_case5[] = { 75,76,77 };
  int pathlist_expected_case6[] = { 75,76,77,78 };
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkLatLonToPathList");

  /* Normal call test case */
  lat =  66.222667343954;
  lon = 110.46694104;

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
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

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
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

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
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

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
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

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
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

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  if (status == MTK_SUCCESS &&
      checkpathlist(pathcnt, pathlist, pathlist_expected_case6)) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }
  free(pathlist);

  /* Failure call test case */
  lat =  85.0;
  lon =  20.0;

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  if (status == MTK_NOT_FOUND) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkLatLonToPathList(lat, lon, NULL, &pathlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkLatLonToPathList(lat, lon, &pathcnt, NULL);
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
