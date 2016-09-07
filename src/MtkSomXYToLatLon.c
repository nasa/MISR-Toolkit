/*===========================================================================
=                                                                           =
=                             MtkSomXYToLatLon                              =
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
  double som_x;          /* SOM X */
  double som_y;          /* SOM Y */
  int disp_rad;          /* Output in Radians */
  int disp_dms;          /* Output in Degrees Minutes Seconds */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  double lat_dd, lon_dd, lat_rad, lon_rad, lat_dms, lon_dms;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkSomXYToLatLon(argr.path,argr.som_x,argr.som_y,&lat_dd,&lon_dd);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkSomXYToLatLon Failed.\n");
  }

  if (argr.disp_rad) {
    MtkDdToRad(lat_dd,&lat_rad);
    MtkDdToRad(lon_dd,&lon_rad);
    printf("Lat: %f  Lon: %f\n",lat_rad,lon_rad);
  }
  else if (argr.disp_dms) {
    MtkDdToDms(lat_dd,&lat_dms);
    MtkDdToDms(lon_dd,&lon_dms);
    printf("Lat: %f  Lon: %f\n",lat_dms,lon_dms);
  }
  else
    printf("Lat: %f  Lon: %f\n",lat_dd,lon_dd);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--path=path> <--somxy=som_x,som_y> [--rad | --dms]\n\n",func);

  fprintf(stderr, "Where: --path=path is the path number\n");
  fprintf(stderr, "       --somxy=som_x,som_y is SomX and SomY\n");
  fprintf(stderr, "       --rad display output in radians\n");
  fprintf(stderr, "       --dms display output in degrees minutes seconds\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkSomXYToLatLon --path=1 --somxy=10529200.016621,622600.018066\n");
  fprintf(stderr, "         MtkSomXYToLatLon --path=1 --somxy=10529200.016621,622600.018066 --rad\n");
  fprintf(stderr, "         MtkSomXYToLatLon --path=1 --somxy=10529200.016621,622600.018066 --dms\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;      /* Return code of this function */
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;
  char *s;
  double som_x = 0.0;
  double som_y = 0.0;

  /* options descriptor */
  static struct option longopts[] = {
    { "path", required_argument, 0, 'p' },
    { "somxy",required_argument, 0, 's' },
    { "rad",  no_argument,       0, 'r' },
    { "dms",  no_argument,       0, 'd' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  argr->disp_rad = 0;
  argr->disp_dms = 0;

  while ((ch = getopt_long(argc, argv, "p:s:rdh", longopts, NULL)) != -1) {

    if (ch != '?' && ch != 'h' && ch != 'p' && ch != 'r' && ch != 'd') {
      optflag = 1;
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid SomX");
      som_x = strtod(s, NULL);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid SomY");
      som_y = strtod(s,NULL);
    }

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'p' : argr->path = (int)atol(optarg);
      break;
    case 's' : argr->som_x = som_x;
      argr->som_y = som_y;
      break;
    case 'r' : argr->disp_rad = 1;
      break;
    case 'd' : argr->disp_dms = 1;
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
  status_code = MTK_FAILURE;
  return status_code;
}
