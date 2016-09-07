/*===========================================================================
=                                                                           =
=                               MtkFillValueGet                             =
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

typedef struct {
  char infile[200];		/* Input HDF filename */
  char grid[200];		  /* Gridname */
  char field[200];		/* Fieldname */
} argr_type;          /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;	/* Data Buffer */
  argr_type argr;               /* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkFillValueGet(argr.infile, argr.grid, argr.field, &dbuf);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkFillValueGet failed!");
  }
  switch (dbuf.datatype)
  {
  case MTKe_void :
    break;
  case  MTKe_char8 :
    printf("FillValue: %c",dbuf.data.c8[0][0]);
    break;
  case  MTKe_uchar8 :
    printf("FillValue: %c",dbuf.data.uc8[0][0]);
    break;
  case  MTKe_int8 :
    printf("FillValue: %d\n",dbuf.data.i8[0][0]);
    break;
  case  MTKe_uint8 :
    printf("FillValue: %u\n",dbuf.data.u8[0][0]);
    break;
  case  MTKe_int16 :
    printf("FillValue: %d\n",dbuf.data.i16[0][0]);
    break;
  case MTKe_uint16 :
    printf("FillValue: %u\n",dbuf.data.u16[0][0]);
    break;
  case  MTKe_int32 :
    printf("FillValue: %d\n",dbuf.data.i32[0][0]);
    break;
  case  MTKe_uint32 :
    printf("FillValue: %u\n",dbuf.data.u32[0][0]);
    break;
  case  MTKe_int64 :
    printf("FillValue: %lld\n",(long long int) dbuf.data.i64[0][0]);
    break;
  case  MTKe_uint64 :
    printf("FillValue: %llu\n",(long long unsigned int) dbuf.data.u64[0][0]);
    break;
  case MTKe_float :
    printf("FillValue: %f\n",dbuf.data.f[0][0]);
    break;
  case  MTKe_double :
    printf("FillValue: %f\n",dbuf.data.d[0][0]);
    break;
  }
  return 0;

ERROR_HANDLE:
  MtkDataBufferFree(&dbuf);
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     --hdffilename=<Input File>\n"
          "     --gridname=<Grid Name>\n"
          "     --fieldname=<Field Name>\n\n",func);

  fprintf(stderr, "Where: --hdffilename=file is a MISR Product File.\n");
  fprintf(stderr, "       --gridname=grid_name is the name of the grid.\n");
  fprintf(stderr, "       --fieldname=field_name is the name of the field.\n\n");

  fprintf(stderr, "\nExample: MtkFillValueGet --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\"\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  int ch;
  int nflag = 0, gflag = 0, fflag = 0;

  /* options descriptor */
  static struct option longopts[] = {
    { "hdffilename",                 required_argument, 0, 'n' },
    { "gridname",                    required_argument, 0, 'g' },
    { "fieldname",                   required_argument, 0, 'f' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "n:g:f:h",
         longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
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
    }
  }

  if (!(nflag && gflag && fflag))  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;
ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
