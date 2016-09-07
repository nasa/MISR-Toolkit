/*===========================================================================
=                                                                           =
=                     MtkPathBlockRangeToBlockCorners                       =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
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
  int path;                     /* Path Number */
  int startblock;               /* Start Block */
  int endblock;                 /* End Block */
} argr_type;                    /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_BlockCorners block_corners = MTKT_BLOCKCORNERS_INIT;
				/* Block corners */
  argr_type argr;               /* Parse arguments */
  int i;

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkPathBlockRangeToBlockCorners(argr.path, argr.startblock, argr.endblock, &block_corners);

  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkPathBlockRangeToBlockCorners failed!");
  }

  for (i = block_corners.start_block; i <= block_corners.end_block; ++i)
  {
    printf("Block: %d\n",block_corners.block[i].block_number);
    printf("  Ulc: (%f,%f)\n",block_corners.block[i].ulc.lat,block_corners.block[i].ulc.lon);
    printf("  Urc: (%f,%f)\n",block_corners.block[i].urc.lat,block_corners.block[i].urc.lon);
    printf("  Ctr: (%f,%f)\n",block_corners.block[i].ctr.lat,block_corners.block[i].ctr.lon);
    printf("  Lrc: (%f,%f)\n\n",block_corners.block[i].lrc.lat,block_corners.block[i].lrc.lon);
    printf("  Llc: (%f,%f)\n\n",block_corners.block[i].llc.lat,block_corners.block[i].llc.lon);
  }

  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     --path=<Path Number>\n"
          "     --startblock=<Start Block>\n"
          "     --endblock=<End Block>\n",func);

  fprintf(stderr, "\nWhere: --path=path_num is the path number.\n");
  fprintf(stderr, "       --startblock=start_block is starting block.\n");
  fprintf(stderr, "       --endblock=end_block is ending block.\n");

  fprintf(stderr, "\nExample 1: MtkPathBlockRangeToBlockCorners --path=37 --startblock=35 --endblock=40\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch;
  int pflag = 0, sflag = 0, eflag = 0;

  /* options descriptor */
  static struct option longopts[] = {

    { "path",                        required_argument, 0, 'p' },
    { "startblock",                  required_argument, 0, 's' },
    { "endblock",                    required_argument, 0, 'e' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "p:s:e:h",
         longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'p':
      argr->path = atoi(optarg);
      pflag = 1;
      break;
    case 's':
      argr->startblock = atoi(optarg);
      sflag = 1;
    case 'e':
      argr->endblock = atoi(optarg);
      eflag = 1;
    }
  }

  if (!(pflag && sflag && eflag))
  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;
ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
