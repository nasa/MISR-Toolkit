/*===========================================================================
=                                                                           =
=                               MtkPixelTime                                =
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
  char infile[200];		/* Input HDF filename */
  double som_x;			/* Som X */
  double som_y;			/* Som Y */
} argr_type;                    /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_TimeMetaData time_metadata = MTKT_TIME_METADATA_INIT;
				/* Time metadata */
  char pixel_time[MTKd_DATETIME_LEN];/* Pixel time */
  argr_type argr;               /* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkTimeMetaRead(argr.infile, &time_metadata);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkTimeMetaRead failed!");
  }

  status = MtkPixelTime(time_metadata, argr.som_x, argr.som_y, pixel_time);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkPixelTime failed!");
  }

  printf("%s\n", pixel_time);

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     --hdffilename=<L1B2 Product File>\n"
          "     --somxy=somx,somy\n\n",func);

  fprintf(stderr, "Where: --hdffilename=file is a MISR L1B2 Product File.\n");
  fprintf(stderr, "       --somxy=som_x,som_y is SomX and SomY.\n\n");

  fprintf(stderr, "\nExample: MtkPixelTime --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --somxy=10529200.016621,622600.018066\n");

}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch, optflag = 0;
  char *s;
  int nflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "hdffilename",                 required_argument, 0, 'n' },
    { "somxy",                       required_argument, 0, 's' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:u:e:l:n:g:f:b:h",
         longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_SUCCESS);
      break;
    case 'n':
      strcpy(argr->infile,optarg);
      nflag = 1;
      break;
    case 's' :
      optflag = 1;
      if ((s = strtok(optarg, ",")) == NULL)
        MTK_ERR_MSG_JUMP("Invalid SomX");
      argr->som_x = strtod(s, NULL);
      if ((s = strtok(NULL, ",")) == NULL)
        MTK_ERR_MSG_JUMP("Invalid SomY");
      argr->som_y = strtod(s,NULL);
      break;
    case '?':
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (!(optflag && nflag))
  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;
ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
