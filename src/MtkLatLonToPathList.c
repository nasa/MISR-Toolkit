/*===========================================================================
=                                                                           =
=                           MtkLatLonToPathList                             =
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
#include <stdio.h>		/* for printf */
#include <stdlib.h>		/* for exit and strtod */
#include <getopt.h>		/* for getopt_long */
#include <string.h>		/* for strtok */

typedef struct {
  double lat_dd;		/* Latitude in decimal degrees */
  double lon_dd;		/* Longitude in decimal degrees */
} argr_type;			/* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  int pathcnt;			/* Path count */
  int *pathlist;		/* Path list */
  int i;			/* Loop index */
  argr_type argr;		/* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkLatLonToPathList(argr.lat_dd, argr.lon_dd, &pathcnt, &pathlist);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkLatLonToPathList failed!\n");
  }

  printf("  Lat/Lon: %f %f\nPath list: ",argr.lat_dd, argr.lon_dd);
  for (i = 0; i < pathcnt; i++) {
    printf("%d ",pathlist[i]);
  }
  printf("\n");

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--dd=lat_dd,lon_dd> | <--rad=lat_r,lon_r> | "
                  "<--dms=lat_dms,lon_dms>\n\n",func);
  fprintf(stderr, "Where: --dd=lat_dd,lon_dd is lat,lon in decimal degrees\n");
  fprintf(stderr, "       --rad=lat_r,lon_r is lat,lon in radians\n");
  fprintf(stderr, "       --dms=lat_dms,lon_dms is lat,lon in packed degrees, "
                  "minutes and seconds\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkLatLonToPathList --dd=-75.345,169.89\n");
  fprintf(stderr, "         MtkLatLonToPathList --rad=-1.315,2.965\n");
  fprintf(stderr, "         MtkLatLonToPathList --dms=-75020042.000,"
                  "169053024.000\n\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;
  double lat = 0.0, lon = 0.0;
  char *s;

  /* options descriptor */
  static struct option longopts[] = {
    { "dd",   required_argument, 0, 'd' },
    { "rad",  required_argument, 0, 'r' },
    { "dms",  required_argument, 0, 'p' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:r:p:h", longopts, NULL)) != -1) {

    if (ch != '?' && ch != 'h') {
      optflag = 1;
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid latitude");
      lat = strtod(s, NULL);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid longitude");
      lon = strtod(s,NULL);
    }

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'd':
      argr->lat_dd = lat;
      argr->lon_dd = lon;
      break;
    case 'r':
      MtkRadToDd(lat, &(argr->lat_dd));
      MtkRadToDd(lon, &(argr->lon_dd));
      break;
    case 'p':
      MtkDmsToDd(lat, &(argr->lat_dd));
      MtkDmsToDd(lon, &(argr->lon_dd));
      break;
    case '?':
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (!optflag) MTK_ERR_MSG_JUMP("Invalid arguments");

  return MTK_SUCCESS;
 ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
