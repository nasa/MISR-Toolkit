/*===========================================================================
=                                                                           =
=                            MtkDmsToDegMinSec                              =
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
  double dms;            /* Packed degrees, minutes, seconds */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  int deg;
  int min;
  double sec;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkDmsToDegMinSec(argr.dms,&deg,&min,&sec);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkDmsToDegMinSec Failed.\n");
  }

  printf("Deg: %d  Min: %d  Sec: %f\n",deg,min,sec);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--dms=packed dms>\n\n",func);

  fprintf(stderr, "Where: --dms=packed dms is packed degrees minutes seconds\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkDmsToDegMinSec --dms=130004058.23\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "dms",  required_argument, 0, 'd' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0  }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'd' : argr->dms = atof(optarg);
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
