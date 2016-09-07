/*===========================================================================
=                                                                           =
=                               MtkFileType                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2013, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrToolkit.h"
#include "MisrError.h"
#include <stdio.h>		/* for printf */
#include <stdlib.h>		/* for exit and strtod */
#include <getopt.h>		/* for getopt_long */
#include <string.h>		/* for strtok */

#define MAX_STR_LEN 200

typedef struct {
  char infile[MAX_STR_LEN];		/* Input HDF filename */
} argr_type;                    /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {
  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  argr_type argr;               /* Parse arguments */
  MTKt_FileType filetype;
  char *filetype_str[] = MTKT_FILE_TYPE_DESC;
  char type_desc[MAX_STR_LEN];

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
  status = MtkFileType(argr.infile, &filetype);
  MTK_ERR_COND_JUMP(status);
  if ( (int)strlen(filetype_str[filetype]) > MAX_STR_LEN )
  		MTK_ERR_CODE_JUMP(MTK_FAILURE);
  strcpy(type_desc,filetype_str[filetype]);
  printf("Filetype: %s\n",type_desc);
  return 0;

ERROR_HANDLE:
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     --hdffilename=<Input File>\n\n",func);

  fprintf(stderr, "Where: --hdffilename=file is a MISR Product File.\n");

  fprintf(stderr, "\nExample: MtkFileType --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {
  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  int ch;
  int nflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "hdffilename",                 required_argument, 0, 'n' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "n:h",
         longopts, NULL)) != -1) {
    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'n':
      if ( (int)strlen(optarg) > MAX_STR_LEN )
		MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);
      strcpy(argr->infile,optarg);
      nflag = 1;
      break;
    }
  }

  if (!(nflag))
  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
