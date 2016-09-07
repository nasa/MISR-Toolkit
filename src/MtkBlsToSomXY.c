/*===========================================================================
=                                                                           =
=                              MtkBlsToSomXY                                =
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
  int block;             /* Block Number */
  float line;            /* Line */
  float sample;          /* Sample */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  double som_x;
  double som_y;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkBlsToSomXY(argr.path, argr.resolution_meters, argr.block,
			 argr.line, argr.sample, &som_x, &som_y);
  printf("SomX: %f  SomY: %f\n",som_x,som_y);
  if (status) {
    status_code = status;
  MTK_ERR_MSG_JUMP("MtkBlsToSomXY Failed.\n");
  }

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--path=path> <--res=resolution> <--bls=block,line,sample>\n\n",func);

  fprintf(stderr, "Where: --path=path is the path number\n");
  fprintf(stderr, "       --res=resolution is the resolution in meters\n");
  fprintf(stderr, "       --bls=block,line,sample is block, line, and sample\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkBlsToSomXY --path=1 --res=1100 --bls=22,101,22\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;
  char *s;

  /* options descriptor */
  static struct option longopts[] = {
    { "path", required_argument, 0, 'p' },
    { "res",  required_argument, 0, 'r' },
    { "bls",  required_argument, 0, 'b' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "p:r:b:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'p': argr->path = (int)atol(optarg);
      break;
    case 'r': argr->resolution_meters = (int)atol(optarg);
      break;
    case 'b':
      optflag = 1;
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid block");
      argr->block = (int)atoi(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid line");
      argr->line = (float)atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid sample");
      argr->sample = (float)atof(s);
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
