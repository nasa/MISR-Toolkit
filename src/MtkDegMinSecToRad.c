/*===========================================================================
=                                                                           =
=                            MtkDegMinSecToRad                              =
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
  int deg;               /* Degrees */
  int min;               /* Minutes */
  double sec;            /* Seconds */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  double rad;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkDegMinSecToRad(argr.deg,argr.min,argr.sec,&rad);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkDegMinSecToRad Failed.\n");
  }

  printf("Rad: %f\n",rad);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--deg=degrees> <--min=minutes> <--sec=seconds>\n\n",func);

  fprintf(stderr, "Where: --deg=degrees is the number of degrees\n");
  fprintf(stderr, "       --min=minutes is the number of minutes\n");
  fprintf(stderr, "       --sec=seconds is the number of seconds\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkDegMinSecToRad --deg=130 --min=4 --sec=58.23\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "deg",  required_argument, 0, 'd' },
    { "min",  required_argument, 0, 'm' },
    { "sec",  required_argument, 0, 's' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0  }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:m:s:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'd' : argr->deg = (int)atoi(optarg);
      break;
    case 'm' : argr->min = (int)atoi(optarg);
      break;
    case 's' : argr->sec = atof(optarg);
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
