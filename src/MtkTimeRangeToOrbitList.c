/*===========================================================================
=                                                                           =
=                         MtkTimeRangeToOrbitList                           =
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
  char *start;           /* Start Time */
  char *end;             /* End Time */
} argr_type;		 /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;		/* Parse arguments */
  int num_orbits;
  int *orbit_list = NULL;
  int i;

  argr.start = NULL;
  argr.end = NULL;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkTimeRangeToOrbitList(argr.start,argr.end,&num_orbits,&orbit_list);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkTimeRangeToOrbitList Failed.\n");
  }

  if (num_orbits == 0)
    printf("No orbits found.\n");
  else {
    printf("Orbit(s):\n");
    for (i = 0; i < num_orbits; ++i)
      printf("%d\n",orbit_list[i]);
  }

  free(argr.start);
  free(argr.end);
  if (orbit_list != NULL)
    free(orbit_list);

  return 0;

ERROR_HANDLE:
  if (argr.start != NULL)
    free(argr.start);

  if (argr.end != NULL)
    free(argr.end);

  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> | <--start=YYYY-MM-DDThh:mm:ssZ> <--end=YYYY-MM-DDThh:mm:ssZ>\n\n",func);

  fprintf(stderr, "Where: --start=YYYY-MM-DDThh:mm:ssZ\n");
  fprintf(stderr, "       --end=YYYY-MM-DDThh:mm:ssZ\n");
  fprintf(stderr, "       --help is this info\n");
  fprintf(stderr, "\nExample: MtkTimeRangeToOrbitList --start=2002-02-02T02:00:00Z --end=2002-05-02T05:00:00Z\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "start",required_argument, 0, 's' },
    { "end",  required_argument, 0, 'e' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0  }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "s:e:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 's' :
      argr->start = (char*)malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->start == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
      strcpy(argr->start,optarg);
      break;
    case 'e' :
      argr->end = (char*)malloc((strlen(optarg) + 1) * sizeof(char));
      if (argr->end == NULL)
	MTK_ERR_CODE_JUMP(MTK_MALLOC_FAILED);
      strcpy(argr->end,optarg);
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
