/*===========================================================================
=                                                                           =
=                            MtkReadBlockRange                              =
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
  char infile[200];		/* Input HDF filename */
  char outfile[200];            /* Output binary filename */
  char grid[200];		/* Gridname */
  char field[200];		/* Fieldname */
  int startblock;               /* Start Block */
  int endblock;                 /* End Block */
} argr_type;                    /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer3D dbuf = MTKT_DATABUFFER3D_INIT; /* Data Buffer */
  argr_type argr;               /* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkReadBlockRange(argr.infile, argr.grid, argr.field, argr.startblock, argr.endblock, &dbuf);

  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkReadBlockRange failed!");
  }

  status = MtkWriteBinFile3D(argr.outfile, dbuf);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkWriteBinFile3D failed!");
  }

  MtkDataBufferFree3D(&dbuf);

  return 0;

ERROR_HANDLE:
  MtkDataBufferFree3D(&dbuf);
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     [--entire-file                                                  |\n"
          "     --hdffilename=<Input File>\n"
          "     --gridname=<Grid Name>\n"
          "     --fieldname=<Field Name>\n"
          "     --startblock=<Start Block>\n"
          "     --endblock=<End Block>\n"
          "     --binfilename=<Binary Output File>\n\n",func);

  fprintf(stderr, "Where: --entire-file queries the hdffile for the block range.\n");
  fprintf(stderr, "       --hdffilename=file is a MISR Product File.\n");
  fprintf(stderr, "       --gridname=grid_name is the name of the grid.\n");
  fprintf(stderr, "       --fieldname=field_name is the name of the field.\n");
  fprintf(stderr, "       --startblock=start_block is starting block.\n");
  fprintf(stderr, "       --endblock=end_block is ending block.\n");
  fprintf(stderr, "       --binfilename=file is the output file.\n\n");

  fprintf(stderr, "\nExample 1: MtkReadBlockRange --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --startblock=35 --endblock=40 --binfilename=out.bin\n");

  fprintf(stderr, "\nExample 2: MtkReadBlockRange --entire-file --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  MTKt_status status;
  extern char *optarg;
  extern int optind;
  int ch, fullflag=0;
  int nflag = 0, gflag = 0, fflag = 0, sflag = 0, eflag = 0, bflag = 0;

  /* options descriptor */
  static struct option longopts[] = {

    { "entire-file",                 no_argument,       0, 'd' },
    { "hdffilename",                 required_argument, 0, 'n' },
    { "gridname",                    required_argument, 0, 'g' },
    { "fieldname",                   required_argument, 0, 'f' },
    { "startblock",                  required_argument, 0, 's' },
    { "endblock",                    required_argument, 0, 'e' },
    { "binfilename",                 required_argument, 0, 'b' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:n:g:f:s:e:h",
         longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'd':
      fullflag = 1;
      break;
    case 'n':
      strcpy(argr->infile,optarg);
      nflag = 1;
      break;
    case 'g':
      strcpy(argr->grid,optarg);
      gflag = 1;
      break;
    case 'f':
      strcpy(argr->field,optarg);
      fflag = 1;
      break;
    case 's':
      argr->startblock = atoi(optarg);
      sflag = 1;
    case 'e':
      argr->endblock = atoi(optarg);
      eflag = 1;
    case 'b':
      strcpy(argr->outfile,optarg);
      bflag = 1;
      break;
    }
  }

  if (fullflag && nflag) {
    status = MtkFileToBlockRange(argr->infile, &argr->startblock,
                                 &argr->endblock);
    MTK_ERR_COND_JUMP(status);
    printf("start block = %d\n",argr->startblock);
    printf("end block = %d\n",argr->endblock);
    sflag = 1;
    eflag = 1;
  }

  if (!(nflag && gflag && fflag && sflag && eflag &&bflag))
  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;
ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
