/*===========================================================================
=                                                                           =
=                                  foo                                      =
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

int bar( double, double );
int biz();

int main() {

  MTKt_status status;
  int result;
  int path = 37;
  int resolution = 275;
  int block = 60;
  float line = 256;
  float sample = 1024;
  double lat_dd, lon_dd;
  int b;
  float l, s;
  int latdeg, londeg, latmin, lonmin;
  double latsec, lonsec;

  status = MtkBlsToLatLon(path, resolution, block, line, sample,
			  &lat_dd, &lon_dd);
  if (status != MTK_SUCCESS) return 1;

  status = MtkLatLonToBls(path, resolution, lat_dd, lon_dd, &b, &l, &s);
  if (status != MTK_SUCCESS) return 1;

  status = MtkDdToDegMinSec(lat_dd, &latdeg, &latmin, &latsec);
  if (status != MTK_SUCCESS) return 1;

  status = MtkDdToDegMinSec(lon_dd, &londeg, &lonmin, &lonsec);
  if (status != MTK_SUCCESS) return 1;

  printf("\nExample 1:\n");
  printf("path = %d\n", path);
  printf("resolution = %d\n", resolution);
  printf("block, line, sample = %d, %6.1f, %6.1f\n", block, line, sample);
  printf("lat_dd, lon_dd = %f, %f\n", lat_dd, lon_dd);
  printf("lat deg, min, sec = %d:%02d:%5.2f\n", latdeg, latmin, latsec);
  printf("lon deg, min, sec = %d:%02d:%5.2f\n", londeg, lonmin, lonsec);
  printf("b, l, s = %d, %6.1f, %6.1f\n", b, l, s);

  printf("\nExample2:\n");
  result = bar(lat_dd, lon_dd);
  if (result) return 1;

  printf("\nExample 3:\n");
  status = biz();
  if (status != MTK_SUCCESS) return 1;

  printf("\nWorked like champ!\n");
  return 0;
}
