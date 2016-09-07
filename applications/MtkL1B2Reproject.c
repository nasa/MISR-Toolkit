/*===========================================================================
=                                                                           =
=                               MtkL1B2Reproject                            =
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
#include <strings.h> 		/* for strncasecmp */
#include <HdfEosDef.h>		/* Definition of GDopen */

#define MAX_FILENAME 5000
#define MAX_FIELDNAME 1000
#define MAX_LENGTH 1000
#define NUMBER_BAND 4
#define NUMBER_CAMERA 9
#define OUTPUT_GRIDNAME "L1B2Reproject"

/* -------------------------------------------------------------------------*/
/* Structure to contain command-line arguments.                             */
/* -------------------------------------------------------------------------*/

typedef struct {
  char *proj_map_file;  /* Projection/map info file */
  char *terrain_file;   /* GRP_TERRAIN file*/
  char *output_basename;   /* Output basename*/
  int  band;            /* Band to process. -1 = all */
} argr_type;	   /* Argument parse result */

#define ARGR_TYPE_INIT {NULL, NULL, NULL, -1}

/* -------------------------------------------------------------------------*/
/* Local function prototypes                                                */
/* -------------------------------------------------------------------------*/

int process_args(int argc, char *argv[], argr_type *argr);

/* -------------------------------------------------------------------------*/
/*  Main program.                                                           */
/* -------------------------------------------------------------------------*/

int main( int argc, char *argv[] ) {
  MTKt_status status;		/* Return status */
  MTKt_status status_code = MTK_FAILURE;  /* Return code of this function */
  argr_type argr = ARGR_TYPE_INIT;  /* Command-line arguments */
  char *band_name[NUMBER_BAND] = {"Blue","Green","Red","NIR"};
  char *camera_name[NUMBER_CAMERA] = {"Df","Cf","Bf","Af","An",
				      "Aa","Ba","Ca","Da"};
  MTKt_GenericMapInfo target_map_info = MTKT_GENERICMAPINFO_INIT;
				/* Target map information. */
  MTKt_GCTPProjInfo target_proj_info = MTKT_GCTPPROJINFO_INIT;
				/* Target projection information.  */
  MTKt_Region region = MTKT_REGION_INIT; /* SOM region containing target map. */
  MTKt_DataBuffer toa_brf_data = MTKT_DATABUFFER_INIT; 
				/* Top-of-atmosphere BRF data */
  MTKt_MapInfo    toa_rad_rdqi_map_info = MTKT_MAPINFO_INIT;
				/* Map information for terrain RDQI.  */
  MTKt_DataBuffer toa_rad_rdqi = MTKT_DATABUFFER_INIT; 
				/* Top-of-atmosphere radiance data quality indicator */ 
  MTKt_DataBuffer toa_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for TOA BRF. */
  MTKt_MapInfo    toa_brf_map_info = MTKT_MAPINFO_INIT;
				/* Map information for terrain-projected radiance.  */
  MTKt_DataBuffer latitude = MTKT_DATABUFFER_INIT; 
				/* Temporary latitude array for reprojection. */
  MTKt_DataBuffer longitude = MTKT_DATABUFFER_INIT; 
				/* Temporary longitude array for reprojection. */
  MTKt_DataBuffer line_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary line coord array for reprojection. */
  MTKt_DataBuffer sample_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary sample coord array for reprojection. */
  MTKt_DataBuffer resampled_toa_brf_data = MTKT_DATABUFFER_INIT; 
				/* Input TOA BRF, resampled to target map. */
  MTKt_DataBuffer resampled_toa_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for final result. */
  int path; 			/* Orbit path number. */
  int camera;  			/* Camera index. */
  int iband; 			/* Loop iterator. */
  int iline; 			/* Loop iterator. */
  int isample; 			/* Loop iterator. */
  char outputfile[MAX_FILENAME];
  int32 fid = FAIL;		/* HDF-EOS file identifier. */
  int32 gid = FAIL;		/* HDF-EOS grid identifier. */
  int32 hstatus; 		/* HDF-EOS status code */
  float64 upleft[2];	        /* min X, max Y corner of target map. */
  float64 lowright[2];	        /* max X, min Y corner of target map. */
  char *dimlist_ul_lr = "YDim,XDim";
				/* HDF-EOS dimension list for origin UL or 
				   origin LR maps.*/
  char *dimlist_ur_ll = "XDim,YDim";
				/* HDF-EOS dimension list for origin UR or 
				   origin LL maps.*/
  char *dimlist; 		/* HDF-EOS dimension list. */
  int32 edge[2];		/* Size of HDF-EOS data to write. */
  char fieldname[MAX_FIELDNAME]; /* Fieldname to read/write. */
  char gridname[MAX_FIELDNAME];  /* Gridname to read. */
  int size_x;			 /* Size of map along X axis. */
  int size_y;			 /* Size of map along Y axis. */
  int32 tile_rank = 2;		 /* HDF tile rank */
  int32 tile_dims[2] = {64,64};	 /* Tile dimensions. */
  int32 comp_code = HDFE_COMP_DEFLATE; /* GZIP compression code. */
  intn comp_parm[1] = {5};	       /* GZIP compression level. */
  float32 fill_float32 = -9999.0;      /* Fill value for float32 fields. */

  /* ------------------------------------------------------------------ */
  /* Parse command-line arguments.                                      */
  /* ------------------------------------------------------------------ */

  if (process_args(argc, argv, &argr))
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  /* ------------------------------------------------------------------ */
  /* Read projection / map information for target area.                 */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfoRead(argr.proj_map_file, &target_proj_info);
  if (status != MTK_SUCCESS) {
    printf("\n\nCheck that projection/map info filename is correct: %s\n",argr.proj_map_file);
    MTK_ERR_MSG_JUMP("Trouble with MtkGCTPProjInfoRead\n");
  }

  status = MtkGenericMapInfoRead(argr.proj_map_file, &target_map_info);
  if (status != MTK_SUCCESS) {
    MTK_ERR_MSG_JUMP("Trouble with MtkGenericMapInfoRead\n");
  }

  /* ------------------------------------------------------------------ */
  /* Get orbit path number of input file.                               */
  /* ------------------------------------------------------------------ */

  status = MtkFileToPath(argr.terrain_file, &path);
  if (status != MTK_SUCCESS) {
      printf("\n\nCheck that input filename is correct: %s\n",argr.terrain_file);
      MTK_ERR_MSG_JUMP("Trouble with MtkFileToPath(input)\n");
  }

  /* ------------------------------------------------------------------ */
  /* Get camera identifier from terrain input file.                     */
  /* ------------------------------------------------------------------ */

  {
    MTKt_DataBuffer attrbuf = MTKT_DATABUFFER_INIT;

    status = MtkFileAttrGet(argr.terrain_file,"Camera",&attrbuf);
    if (status != MTK_SUCCESS) {
      printf("\n\nCheck that GRP_TERRAIN filename is correct: %s\n",argr.terrain_file);
      MTK_ERR_MSG_JUMP("Trouble with MtkFileAttrGet(terrain_file)\n");
    }
    
    camera = attrbuf.data.i32[0][0] - 1;
  }

  /* ------------------------------------------------------------------ */
  /* Setup SOM region containing the target map.                        */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByGenericMapInfo(&target_map_info,
					&target_proj_info,
					path,
					&region);
  if (status != MTK_SUCCESS) {
    MTK_ERR_MSG_JUMP("Trouble with MtkSetRegionByGenericMapInfo\n");
  }

  /* ------------------------------------------------------------------ */
  /* Create HDF-EOS file for result.                                    */
  /* ------------------------------------------------------------------ */

  if (target_map_info.origin_code == MTKe_ORIGIN_UL ||
      target_map_info.origin_code == MTKe_ORIGIN_LR) {
    dimlist = dimlist_ul_lr;
  } else {
    dimlist = dimlist_ur_ll;
  }
  
  edge[0] = target_map_info.size_line;
  edge[1] = target_map_info.size_sample;

  upleft[0] = target_map_info.min_x;
  upleft[1] = target_map_info.max_y;
  lowright[0] = target_map_info.max_x;
  lowright[1] = target_map_info.min_y;

  size_x = (int)floor((target_map_info.max_x - target_map_info.min_x) / 
		      target_map_info.resolution_x + 0.5);
  size_y = (int)floor((target_map_info.max_y - target_map_info.min_y) / 
		      target_map_info.resolution_y + 0.5);
	
  snprintf(outputfile,MAX_FILENAME,"%s_%s.hdf",argr.output_basename,
	   camera_name[camera]);
  
  fid = GDopen(outputfile,DFACC_CREATE);
  if (fid == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDopen\n");
  }
	
  hstatus = GDcreate(fid,OUTPUT_GRIDNAME,size_x, size_y, 
		     upleft, lowright);
  
  gid = GDattach(fid, OUTPUT_GRIDNAME);
  if (gid == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDattach\n");
  }
  
  hstatus = GDdeforigin(gid,target_map_info.origin_code);
  if (hstatus == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDdeforigin\n");
  }
  
  hstatus = GDdefpixreg(gid,target_map_info.pix_reg_code);
  if (hstatus == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDdefpixreg\n");
  }

  hstatus = GDdefproj(gid,target_proj_info.proj_code,
		      target_proj_info.zone_code, target_proj_info.sphere_code,
		      target_proj_info.proj_param);
  if (hstatus == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDdefproj\n");
  }

  /* ------------------------------------------------------------------ */
  /* Calculate pixel lat/lon locations for target map.                  */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPCreateLatLon(&target_map_info,
			       &target_proj_info,
			       &latitude,
			       &longitude);
  if (status != MTK_SUCCESS) {
    MTK_ERR_MSG_JUMP("Trouble with MtkGCTPCreateLatLon\n");
  }

  /* ------------------------------------------------------------------ */
  /* For each band to process...                                        */
  /* ------------------------------------------------------------------ */

  for (iband = 0 ; iband < NUMBER_BAND ; iband++)  {
    if (argr.band < 0 || iband == argr.band) {

      printf("Processing band %d (%s)...\n",iband, band_name[iband]);

  /* ------------------------------------------------------------------ */
  /* Read TOA BRF for the target area.                                  */
  /* ------------------------------------------------------------------ */

      snprintf(gridname,MAX_FIELDNAME,"%sBand",band_name[iband]);
      snprintf(fieldname,MAX_FIELDNAME,"%s BRF",band_name[iband]);
      
      status = MtkReadData(argr.terrain_file, gridname, fieldname,
		       region, &toa_brf_data, &toa_brf_map_info);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkReadData(Terrain)\n");
      }

  /* ------------------------------------------------------------------ */
  /* Read terrain-projected radiance data quality indicators for the    */
  /* the target area.                                                   */
  /* ------------------------------------------------------------------ */

      snprintf(gridname,MAX_FIELDNAME,"%sBand",band_name[iband]);
      snprintf(fieldname,MAX_FIELDNAME,"%s RDQI",band_name[iband]);
      
      status = MtkReadData(argr.terrain_file, gridname, fieldname,
		       region, &toa_rad_rdqi, &toa_rad_rdqi_map_info);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkReadData(Terrain)\n");
      }

  /* ------------------------------------------------------------------ */
  /* Generate mask indicating valid TOA BRF.                            */
  /* TOA BRF is considered valid where the terrain-projected radiance   */
  /* data quality is less than 2.                                       */
  /* ------------------------------------------------------------------ */

      status = MtkDataBufferAllocate(toa_brf_data.nline, toa_brf_data.nsample,
				     MTKe_uint8,&toa_brf_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkDataBufferAllocate(toa_brf_mask)\n");
      }

      for (iline = 0; iline < toa_brf_data.nline; iline++) {
	for (isample = 0; isample < toa_brf_data.nsample; isample++) {
	  if (toa_rad_rdqi.data.u8[iline][isample] < 2) {
	    toa_brf_mask.data.u8[iline][isample] = 1;
	  } else {
	    toa_brf_mask.data.u8[iline][isample] = 0;
	  }
	}
      }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */
      
      MtkDataBufferFree(&toa_rad_rdqi);

  /* ------------------------------------------------------------------ */
  /* Reproject TOA BRF to target map.                                   */
  /* ------------------------------------------------------------------ */

      status = MtkTransformCoordinates(toa_brf_map_info, 
				       latitude, 
				       longitude,
				       &line_coords,
				       &sample_coords);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkTransformCoordinates\n");
      }

      status = MtkResampleCubicConvolution(&toa_brf_data, 
					   &toa_brf_mask, 
					   &line_coords, 
					   &sample_coords, 
					   -0.5,
					   &resampled_toa_brf_data,
					   &resampled_toa_brf_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkResampleCubicConvolution\n");
      }

  /* ------------------------------------------------------------------ */
  /* Free memory                                                        */
  /* ------------------------------------------------------------------ */
      
      MtkDataBufferFree(&resampled_toa_brf_mask);
      MtkDataBufferFree(&toa_brf_mask);
      MtkDataBufferFree(&toa_brf_data);

  /* ------------------------------------------------------------------ */
  /* Free memory                                                        */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&line_coords);
      MtkDataBufferFree(&sample_coords);

  /* ------------------------------------------------------------------ */
  /* Write input TOA BRF to HDF-EOS file.                               */
  /* ------------------------------------------------------------------ */

      snprintf(fieldname,MAX_FIELDNAME,"TOA_BRF_%s_%s",
	       camera_name[camera], band_name[iband]);

      hstatus = GDdeffield(gid, fieldname, dimlist, DFNT_FLOAT32, 0);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDdeffield\n");
      }

      hstatus = GDsetfillvalue(gid, fieldname, &fill_float32);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDsetfillvalue()\n");
      }
      
      hstatus = GDsettilecomp(gid, fieldname, tile_rank, 
			      tile_dims, comp_code, comp_parm);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDsettilecomp()\n");
      }

      hstatus = GDwritefield(gid, fieldname, NULL, NULL, edge, 
			     resampled_toa_brf_data.dataptr);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDwritefield\n");
      }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&resampled_toa_brf_data);

  /* ------------------------------------------------------------------ */
  /* End loop for each band to process.                                 */
  /* ------------------------------------------------------------------ */

    }
  }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */
  
  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);

  /* ------------------------------------------------------------------ */
  /* Close HDF-EOS file.                                                */
  /* ------------------------------------------------------------------ */

  hstatus = GDdetach(gid);
  if (hstatus == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDdetach\n");
  }
  gid = FAIL;
  
  hstatus = GDclose(fid);
  if (hstatus == FAIL) {
    MTK_ERR_MSG_JUMP("Trouble with GDclose\n");
  }
  fid = FAIL;

  printf("Wrote output to %s\n",outputfile);
  printf("Completed normally.\n");
  return 0;

ERROR_HANDLE:
  if (gid != FAIL) {
    hstatus = GDdetach(gid);
  }
  if (fid != FAIL) {
    hstatus = GDclose(fid);
  }
  
  MtkDataBufferFree(&resampled_toa_brf_mask);
  MtkDataBufferFree(&resampled_toa_brf_data);
  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);
  MtkDataBufferFree(&toa_brf_mask);
  MtkDataBufferFree(&toa_rad_rdqi);
  MtkDataBufferFree(&toa_brf_data);
  MtkDataBufferFree(&line_coords);
  MtkDataBufferFree(&sample_coords);

  printf("Failed: status code = %d\n",status_code);
  
  return status_code;
}

void usage(char *func) {
  fprintf(stderr,
"\nUsage: %s [--help] [--band=<band number>] \n"
"                     <projection/map info file> <GRP_TERRAIN file>\n"
"                     <output basename>\n",
	  func);

  fprintf(stderr,
"\n"
"Perform reproject L1B2 terrain data to given map projection.\n"
"\n"
"Result is written to an HDF-EOS file in the following fields:\n"
"    TOA_BRF_<cam>_<band>             Input TOA BRF from GRP_TERRAIN file (reprojected).\n"
"\n"
"All output fields have identical map projection and size.\n"
"\n"
	  );

  fprintf(stderr,
"COMMAND-LINE OPTIONS\n"
"\n"
"--help\n"
"     Returns this usage info.\n"
"\n"
"--band=<band number>\n"
"     Specifies the band to process.  Bands are identified by integer values as\n"
"     follows: 0 = blue   1 = green   2 = red   3 = nir\n"
"\n"
	  );

  fprintf(stderr,
"COMMAND-LINE ARGUMENTS\n"
"\n"
"<projection/map info file>\n"
"    Text file specifying map projection for the output.  Parameters are\n"
"    specified as name = value pairs.   Parameter names are as follows:\n"
"\n"
"       proj_code is the GCTP projection code.\n"
"       utm_zone is the UTM zone number for UTM projections only.\n"
"       sphere_code is GCTP spheroid code.\n"
"       proj_param(n) is the nth GCTP projection parameter.  (1 <= n <= 15)\n"
"       min_corner_x is the minimum X coordinate at the edge of the map.\n"
"       min_corner_y is the minimum Y coordinate at the edge of the map.\n"
"       resolution_x is the size of a pixel along the X-axis.\n"
"       resolution_y is the size of a pixel along the Y-axis.\n"
"       number_pixel_x is the size of the map in pixels, along the X-axis.\n"
"       number_pixel_y is the size of the map in pixels, along the Y-axis.\n"
"       origin_code defines the corner of the map at which pixel 0,0 is located.\n"
"       pix_reg_code defines whether a pixel value is related to the corner or\n"
"         center of the corresponding area of that pixel on the map.  If the\n"
"         corner is used, then it is always the corner corresponding to the\n"
"         corner of the origin.\n"
"\n"
"       Possible values for origin_code are:\n"
"         UL - Upper Left (min X, max Y);  Line=Y, Sample=X\n"
"         UR - Upper Right (max X, max Y); Line=X, Sample=Y\n"
"         LL - Lower Left (min X, min Y);  Line=X, Sample=Y\n"
"         LR - Lower Right (max X, min Y); Line=Y, Sample=X\n"
"\n"
"       Possible values for pix_reg_code are:\n"
"         CENTER - Center\n"
"         CORNER - Corner\n"
"\n"
"       Unrecognized parameter names are ignored.\n"
"       Lines starting with a '#' character are ignored.\n"
"       Anything after the name = value pair on a line is ignored.\n"
"\n"
"       Example projection/map info file:\n"
"       # Albers equal-area conic projection parameters\n"
"       proj_code = 3    # Albers equal-area conic\n"
"       sphere_code = 8  # GRS80\n"
"       proj_param(3) = 29030000.0  # Latitude of the first standard parallel\n"
"       proj_param(4) = 45030000.0  # Latitude of the second standard parallel\n"
"       proj_param(5) = -96000000.0 # Longitude of the central meridian\n"
"       proj_param(6) = 23000000.0  # Latitude of the projection origin\n"
"       # Map information\n"
"       min_corner_x = 1930612.449614\n"
"       min_corner_y = 2493633.488881\n"
"       resolution_x = 250.0\n"
"       resolution_y = 250.0\n"
"       number_pixel_x = 1311\n"
"       number_pixel_y = 2078\n"
"       origin_code = UL\n"
"       pix_reg_code = CENTER\n"
"\n"
"\n"
"<GRP_TERRAIN>\n"
"    MISR Terrain-projected Radiance Product (GRP_TERRAIN).\n"
"\n"
"<output basename>\n"
"    Basename for output file.\n"
"\n"
"Examples:\n"
"\n"
"     MtkL1B2Reproject --band=2 proj_map_info.txt \\\n"
"         MISR_AM1_GRP_TERRAIN_GM_P011_O013824_DF_F03_0024.b52-56.hdf \\\n"
"         BRF_O013824_maine\n"
"\n"
"     MtkL1B2Reproject \\\n"
"         proj_map_info.txt \\\n"
"         MISR_AM1_GRP_TERRAIN_GM_P011_O013824_DF_F03_0024.b52-56.hdf \\\n"
"         BRF_O013824_maine\n"
"\n"
	  );
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch;
  argr->band = -1;
  
  /* options descriptor */
  static struct option longopts[] = {
    { "band", required_argument, 0, 'b' },
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  while ((ch = getopt_long(argc, argv, "b:h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'b' : 
      argr->band = (int)atol(optarg);
      break;
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (argc-optind != 3) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  argr->proj_map_file = argv[optind++];
  argr->terrain_file = argv[optind++];
  argr->output_basename = argv[optind++];

  return MTK_SUCCESS;
 ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}

