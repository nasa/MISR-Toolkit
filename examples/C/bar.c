/*===========================================================================
=                                                                           =
=                                  bar                                      =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>

int bar( double lat, double lon ) {

  MTKt_status status;
  int pathcnt;
  int *pathlist;
  int i, j;
  int orbitcnt;
  int *orbitlist;
                                           /* YYYY-MM-DDThhmmssZ */
  char *starttime = "2002-02-02T02:00:00Z"; /* 2002-02-02 02:00:00 UTC */
  char *endtime = "2002-05-02T02:00:00Z";   /* 2002-05-02 02:00:00 UTC */

  printf("starttime = %s\nendtime = %s\n", starttime, endtime);

  status = MtkLatLonToPathList(lat, lon, &pathcnt, &pathlist);
  if (status != MTK_SUCCESS) return 1;

  printf("Pathlist = ");
  for (i = 0; i < pathcnt; i++) {
    printf("%d ", pathlist[i]);
  }
  printf("\n");

  for (i = 0; i < pathcnt; i++) {
    status = MtkPathTimeRangeToOrbitList(pathlist[i], starttime, endtime,
					 &orbitcnt, &orbitlist);
    if (status != MTK_SUCCESS) return 1;
    printf("Orbitlist for Path %d = ", pathlist[i]);
    for (j = 0; j < orbitcnt; j++) {
      printf("%d ", orbitlist[j]);
    }
    printf("\n");
  }
  return 0;
}
