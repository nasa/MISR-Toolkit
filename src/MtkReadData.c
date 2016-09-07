/*===========================================================================
=                                                                           =
=                               MtkReadData                                 =
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
  char outfile[200];		/* Output binary filename */
  char grid[200];		/* Gridname */
  char field[200];		/* Fieldname */
  MTKt_Region region;		/* Region to read */
} argr_type;                    /* Argument parse result */

int process_args(int argc, char *argv[], argr_type *argr);

int main( int argc, char *argv[] ) {

  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer dbuf = MTKT_DATABUFFER_INIT;	/* Data Buffer */
  MTKt_MapInfo mapinfo;		/* Map Information */
  argr_type argr;               /* Parse arguments */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkReadData(argr.infile, argr.grid, argr.field, argr.region,
		      &dbuf, &mapinfo);

  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkReadData failed!");
  }

  status = MtkWriteBinFile(argr.outfile, dbuf, mapinfo);
  if (status) {
    status_code = status;
    MTK_ERR_MSG_JUMP("MtkWriteBinFile failed!");
  }

  MtkDataBufferFree(&dbuf);

  return 0;

ERROR_HANDLE:
  MtkDataBufferFree(&dbuf);
  return status_code;
}

void usage(char *func) {
  fprintf(stderr, "\nUsage: %s <--help> |\n"
          "     [--entire-file                                                  |\n"
          "      --setregion-path-blockrange=<path,start_blk,end_blk>           |\n"
          "      --setregion-ulclrc=<ulclat,ulclon,lrclat,lrclon>               |\n"
          "      --setregion-latlon-extent=<lat,lon,latext,lonext,extent_units> ]\n"
          "     --hdffilename=<Input File>\n"
          "     --gridname=<Grid Name>\n"
          "     --fieldname=<Field Name>\n"
          "     --binfilename=<Binary Output File>\n\n",func);

  fprintf(stderr, "Where: --entire-file queries the hdffile for path and block range.\n");
  fprintf(stderr, "       --setregion-path-blockrange=path,start_blk,end_blk is path, start block, end block.\n");
  fprintf(stderr, "       --setregion-ulclrc=ulclat,ulclon,lrclat,lrclon is Upper Left Corner Lat, Lon and Lower Right Corner Lat and Lon.\n");
  fprintf(stderr, "       --setregion-latlon-extents=lat,lon,latext,lonext,extent_units is Lat, Lon in degrees and Extent in specified units.\n");

  fprintf(stderr, "       --hdffilename=file is a MISR Product File.\n");
  fprintf(stderr, "       --gridname=grid_name is the name of the grid.\n");
  fprintf(stderr, "       --fieldname=field_name is the name of the field.\n");
  fprintf(stderr, "       --binfilename=file is the output file.\n\n");

  fprintf(stderr, "Note: The parameter extent_units is a case insensitive string that can be set to one of the following values:\n\n");

  fprintf(stderr, "      1) degrees, deg, dd for degrees;\n");
  fprintf(stderr, "      2) meters, m for meters;\n");
  fprintf(stderr, "      3) kilometers, km for kilometers; and\n");
  fprintf(stderr, "      4) 275m, 275 meters, 1.1km, 1.1 kilometers for pixels of a specified resolution per pixel.\n"); 

  fprintf(stderr, "\nExample 1: MtkReadData --setregion-path-blockrange=37,45,75 --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n");

  fprintf(stderr, "\nExample 2: MtkReadData --setregion-latlon-extent=38,-111,3000,300,km --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n");

  fprintf(stderr, "\nExample 3: MtkReadData --setregion-latlon-extent=38,-111,2000,300,1100m --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n");

  fprintf(stderr, "\nExample 4: MtkReadData --setregion-ulclrc=51.5,-112,24,-109 --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n");

  fprintf(stderr, "\nExample 5: MtkReadData --entire-file --hdffilename=../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf --gridname=BlueBand --fieldname=\"Blue Radiance/RDQI\" --binfilename=out.bin\n\n");
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  MTKt_status status;
  extern char *optarg;
  extern int optind;
  int ch, fullflag=0, optflag = 0;
  char *s;
  int nflag = 0, gflag = 0, fflag = 0, bflag = 0;
  double ulc_lat_dd;
  double ulc_lon_dd;
  double lrc_lat_dd;
  double lrc_lon_dd;
  double ctr_lat_dd;
  double ctr_lon_dd;
  double lat_extent;
  double lon_extent;
  char extent_units[80];
  int path_number;
  int start_block;
  int end_block;

  /* options descriptor */
  static struct option longopts[] = {
    { "entire-file",                 no_argument,       0, 'd' },
    { "setregion-ulclrc",            required_argument, 0, 'u' },
    { "setregion-latlon-extent",     required_argument, 0, 'e' },
    { "setregion-path-blockrange",   required_argument, 0, 'l' },
    { "hdffilename",                 required_argument, 0, 'n' },
    { "gridname",                    required_argument, 0, 'g' },
    { "fieldname",                   required_argument, 0, 'f' },
    { "binfilename",                 required_argument, 0, 'b' },
    { "help",                        no_argument,       0, 'h' },
    { 0, 0, 0, 0 }
  };

  if (argc == 1) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  while ((ch = getopt_long(argc, argv, "d:u:e:l:n:g:f:b:h",
         longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'd':
      fullflag = 1;
      break;
    case 'u':
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid ULC Lat");
      ulc_lat_dd = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid ULC Lon");
      ulc_lon_dd = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid LRC Lat");
      lrc_lat_dd = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid LRC Lon");
      lrc_lon_dd = atof(s);
      status = MtkSetRegionByUlcLrc(ulc_lat_dd, ulc_lon_dd,
				    lrc_lat_dd, lrc_lon_dd,
				    &argr->region);
      MTK_ERR_COND_JUMP(status);
      optflag = 1;
      break;
    case 'e':
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Center Lat");
      ctr_lat_dd = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Center Lon");
      ctr_lon_dd = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Lat Extent");
      lat_extent = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Lon Extent");
      lon_extent = atof(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Extent Units");
      strcpy(extent_units, s);
      status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
					  lat_extent,
					  lon_extent,
					  (const char *)extent_units,
					  &argr->region);
      MTK_ERR_COND_JUMP(status);
      optflag = 1;
      break;
    case 'l':
      if ((s = strtok(optarg, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Path");
      path_number = atoi(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid Start Block");
      start_block = atoi(s);
      if ((s = strtok(NULL, ",")) == NULL)
	MTK_ERR_MSG_JUMP("Invalid End Block");
      end_block = atoi(s);
      status = MtkSetRegionByPathBlockRange(path_number, start_block,
					    end_block, &argr->region);
      MTK_ERR_COND_JUMP(status);
      optflag = 1;
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
    case 'b':
      strcpy(argr->outfile,optarg);
      bflag = 1;
      break;
    }
  }

  if (fullflag && nflag) {
    status = MtkFileToPath(argr->infile, &path_number);
    MTK_ERR_COND_JUMP(status);
    status = MtkFileToBlockRange(argr->infile, &start_block, &end_block);
    MTK_ERR_COND_JUMP(status);
    printf("path = %d\n",path_number);
    printf("start block = %d\n",start_block);
    printf("end block = %d\n",end_block);
    status = MtkSetRegionByPathBlockRange(path_number, start_block,
					  end_block, &argr->region);
    MTK_ERR_COND_JUMP(status);
    optflag = 1;
  }

  if (!(optflag && nflag && gflag && fflag && bflag))
  {
    status_code = MTK_BAD_ARGUMENT;
    MTK_ERR_MSG_JUMP("Invalid arguments");
  }

  return MTK_SUCCESS;
ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}
