/*===========================================================================
=                                                                           =
=                               MtkSomXYToBls                               =
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
  double som_x;          /* SOM X */
  double som_y;          /* SOM Y */
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



  status = MtkSomXYToBls(argr.path,argr.resolution_meters,argr.som_x,
			 argr.som_y,&block,&line,&sample);

  printf("Block: %d  Line: %f  Sample: %f\n",block,line,sample);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkSomXYToBls Failed.\n");
  }

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--path=path> <--res=resolution> <--somxy=som_x,som_y>\n\n",func);

  fprintf(stderr, "Where: --path=path is the path number\n");
  fprintf(stderr, "       --res=resolution is the resolution in meters\n");
  fprintf(stderr, "       --somxy=som_x,som_y is SomX and SomY\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkSomXYToBls --path=1 --res=1100 --somxy=10529200.016621,622600.018066\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE; /* Return code of this function */
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;
  double som_x = 0.0, som_y = 0.0;
  char *s;

  /* options descriptor */
  static struct option longopts[] = {
    { "path", required_argument, 0, 'p' },
    { "res",  required_argument, 0, 'r' },
    { "somxy",required_argument, 0, 's' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "p:r:s:h", longopts, NULL)) != -1) {

    if (ch != '?' && ch != 'h' && ch != 'p' && ch != 'r') {
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
    case 'r' :
      argr->resolution_meters = (int)atol(optarg);
      break;
    case 's' :
      argr->som_x = som_x;
      argr->som_y = som_y;
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
