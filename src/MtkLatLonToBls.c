/*===========================================================================
=                                                                           =
=                               MtLatLonToBls                               =
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
#include <getopt.h>		/* for getopt_long */

typedef struct {
  int path;              /* Path */
  int resolution_meters; /* Resolution */
  double lat_dd;         /* Latitude */
  double lon_dd;         /* Longitude */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  int block;
  float line;
  float sample;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkLatLonToBls(argr.path, argr.resolution_meters, argr.lat_dd,
                          argr.lon_dd, &block, &line, &sample);
  printf("Block: %d  Line: %f  Sample: %f\n",block,line,sample);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkLatLonToBls Failed.\n");
  }

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--path=path> <--res=resolution> [<--dd=lat_dd,lon_dd> | <--rad=lat_r,lon_r> | "
                  "<--dms=lat_dms,lon_dms>]\n\n",func);
  fprintf(stderr, "Where: --path=path is the path number\n");
  fprintf(stderr, "       --res=resolution is the resolution in meters\n");
  fprintf(stderr, "       --dd=lat_dd,lon_dd is lat,lon in decimal degrees\n");
  fprintf(stderr, "       --rad=lat_r,lon_r is lat,lon in radians\n");
  fprintf(stderr, "       --dms=lat_dms,lon_dms is lat,lon in packed degrees, "
                  "minutes and seconds\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkLatLonToBls --path=1 --res=1100 --dd=82.740690,-3.310459\n");
  fprintf(stderr, "         MtkLatLonToBls --path=1 --res=1100 --rad=1.444098,-0.057778\n");
  fprintf(stderr, "         MtkLatLonToBls --path=1 --res=1100 --dms=82044026.490000,-3018037.650000\n");
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
    { "path", required_argument, 0, 'p' },
    { "res",  required_argument, 0, 'r' },
    { "dd",   required_argument, 0, 'd' },
    { "rad",  required_argument, 0, 'a' },
    { "dms",  required_argument, 0, 'm' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "p:r:d:a:m:h", longopts, NULL)) != -1) {

    if (ch != '?' && ch != 'h' && ch != 'p' && ch != 'r') {
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
    case 'p' : argr->path = (int)atol(optarg);
      break;
    case 'r' : argr->resolution_meters = (int)atol(optarg);
      break;
    case 'd' :
      argr->lat_dd = lat;
      argr->lon_dd = lon;
      break;
    case 'a' :
      MtkRadToDd(lat, &(argr->lat_dd));
      MtkRadToDd(lon, &(argr->lon_dd));
      break;
    case 'm' :
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
