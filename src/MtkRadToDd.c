/*===========================================================================
=                                                                           =
=                                MtkRadToDd                                 =
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
  double rad;            /* Radians */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  double dd;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkRadToDd(argr.rad,&dd);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkRadToDd Failed.\n");
  }

  printf("Dd: %f\n",dd);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--rad=radians>\n\n",func);

  fprintf(stderr, "Where: --rad=radians is the number of radians\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkRadToDd --rad=2.270373886\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE; /* Return code of this function */
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "rad",  required_argument, 0, 'r' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0  }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "r:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'r' : argr->rad = atof(optarg);
      optflag = 1;
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
