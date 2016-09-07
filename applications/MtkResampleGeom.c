/*===========================================================================
=                                                                           =
=                               MtkResampleGeom                             =
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
#define GP_GMP_GRID_NAME "GeometricParameters"
#define OUTPUT_GRIDNAME "GeometricParameters"
#define NUMBER_FIELD 20
#define NUMBER_CAMERA 9

/* -------------------------------------------------------------------------*/
/* Structure to contain command-line arguments.                             */
/* -------------------------------------------------------------------------*/

typedef struct {
  char *proj_map_file;  /* Projection/map info file */
  char *geom_file;      /* AS_LAND file */
  char *output_filename; /* Output filename*/
} argr_type;	   /* Argument parse result */

#define ARGR_TYPE_INIT {NULL, NULL, NULL}

/* -------------------------------------------------------------------------*/
/* Local function prototypes                                                */
/* -------------------------------------------------------------------------*/

int process_args(int argc, char *argv[], argr_type *argr);

/* -------------------------------------------------------------------------*/
/*  Main program.                                                           */
/* -------------------------------------------------------------------------*/

int main( int argc, char *argv[] ) {
  MTKt_status status;		/* Return status */
  MTKt_status status_code = MTK_FAILURE;      /* Return code of this function */
  argr_type argr = ARGR_TYPE_INIT;  /* Command-line arguments */
  MTKt_GenericMapInfo target_map_info = MTKT_GENERICMAPINFO_INIT;
				/* Target map information. */
  MTKt_GCTPProjInfo target_proj_info = MTKT_GCTPPROJINFO_INIT;
				/* Target projection information.  */
  MTKt_Region region = MTKT_REGION_INIT; /* SOM region containing target map. */
  MTKt_DataBuffer latitude = MTKT_DATABUFFER_INIT; 
				/* Temporary latitude array for reprojection. */
  MTKt_DataBuffer longitude = MTKT_DATABUFFER_INIT; 
				/* Temporary longitude array for reprojection. */
  MTKt_DataBuffer line_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary line coord array for reprojection. */
  MTKt_DataBuffer sample_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary sample coord array for reprojection. */
  MTKt_DataBuffer geom_data = MTKT_DATABUFFER_INIT; 
				/* Data for geometric parameters field. */
  MTKt_DataBuffer geom_data_float64 = MTKT_DATABUFFER_INIT; 
				/* Data for geometric parameters field. */
  MTKt_DataBuffer resampled_geom_data = MTKT_DATABUFFER_INIT; 
				/* Data for geometric parameters field. */
  int path; 			/* Orbit path number. */
  int iline; 			/* Loop iterator. */
  int isample; 			/* Loop iterator. */
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
  char *fieldnames[NUMBER_FIELD] = {"SolarAzimuth",
				    "SolarZenith",
				    "DfAzimuth",
				    "DfZenith",
				    "CfAzimuth",
				    "CfZenith",
				    "BfAzimuth",
				    "BfZenith",
				    "AfAzimuth",
				    "AfZenith",
				    "AnAzimuth",
				    "AnZenith",
				    "AaAzimuth",
				    "AaZenith",
				    "BaAzimuth",
				    "BaZenith",
				    "CaAzimuth",
				    "CaZenith",
				    "DaAzimuth",
				    "DaZenith"};
				/* List of fields to copy. */
  int ifield; 			/* Loop iterator. */
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
    MTK_ERR_MSG_JUMP("Trouble with MtkGCTPProjInfoRead\n");
  }

  status = MtkGenericMapInfoRead(argr.proj_map_file, &target_map_info);
  if (status != MTK_SUCCESS) {
    MTK_ERR_MSG_JUMP("Trouble with MtkGenericMapInfoRead\n");
  }

  /* ------------------------------------------------------------------ */
  /* Get orbit path number of input file.                               */
  /* ------------------------------------------------------------------ */

  MtkFileToPath(argr.geom_file, &path);

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

  fid = GDopen(argr.output_filename,DFACC_CREATE);
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
  /* For each field to copy...                                          */
  /* ------------------------------------------------------------------ */
    
  for (ifield = 0 ; ifield < NUMBER_FIELD; ifield++) {
    MTKt_MapInfo    map_info = MTKT_MAPINFO_INIT;
				/* Map information for this field. */

    printf("Processing %s...\n",fieldnames[ifield]);
 
  /* ------------------------------------------------------------------ */
  /* Read data for the target area.                                     */
  /* ------------------------------------------------------------------ */

    status = MtkReadData(argr.geom_file, GP_GMP_GRID_NAME, fieldnames[ifield],
			 region, &geom_data_float64, &map_info);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkReadData(geom)\n");
    }

  /* ------------------------------------------------------------------ */
  /* Convert float64 to float32.                                        */
  /* ------------------------------------------------------------------ */

    status = MtkDataBufferAllocate(geom_data_float64.nline, geom_data_float64.nsample,
				   MTKe_float, &geom_data);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkDataBufferAllocate(geom_data)\n");
    }
    
    for (iline = 0; iline < geom_data.nline; iline++) {
      for (isample = 0; isample < geom_data.nsample; isample++) {
	geom_data.data.f[iline][isample] = geom_data_float64.data.d[iline][isample];
      }
    }

  /* ------------------------------------------------------------------ */
  /* Reproject data to the target map.                                  */ 
  /* ------------------------------------------------------------------ */

    status = MtkTransformCoordinates(map_info, 
				     latitude, 
				     longitude,
				     &line_coords,
				     &sample_coords);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkTransformCoordinates\n");
    }

    status = MtkResampleNearestNeighbor(geom_data, 
					line_coords, 
					sample_coords, 
					&resampled_geom_data);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkResampleNearestNeighbor\n");
    }

  /* ------------------------------------------------------------------ */
  /* Write reprojected data to HDF-EOS file.                            */
  /* ------------------------------------------------------------------ */
      
    hstatus = GDdeffield(gid, fieldnames[ifield], dimlist, DFNT_FLOAT32, 0);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDdeffield()\n");
    }

    hstatus = GDsetfillvalue(gid, fieldnames[ifield], &fill_float32);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDsetfillvalue()\n");
    }
    
    hstatus = GDsettilecomp(gid, fieldnames[ifield], tile_rank, 
			    tile_dims, comp_code, comp_parm);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDsettilecomp()\n");
    }

    hstatus = GDwritefield(gid, fieldnames[ifield], NULL, NULL, edge, 
			   resampled_geom_data.dataptr);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDwritefield()\n");
    }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

    MtkDataBufferFree(&line_coords);
    MtkDataBufferFree(&sample_coords);
    MtkDataBufferFree(&geom_data);
    MtkDataBufferFree(&geom_data_float64);
    MtkDataBufferFree(&resampled_geom_data);
    
  /* ------------------------------------------------------------------ */
  /* End loop for each field to copy.                                   */
  /* ------------------------------------------------------------------ */

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

  printf("Wrote output to %s\n",argr.output_filename);
  printf("Completed normally.\n");
  return 0;

ERROR_HANDLE:
  if (gid != FAIL) {
    hstatus = GDdetach(gid);
  }
  if (fid != FAIL) {
    hstatus = GDclose(fid);
  }
  
  MtkDataBufferFree(&geom_data);
  MtkDataBufferFree(&geom_data_float64);
  MtkDataBufferFree(&resampled_geom_data);
  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);
  MtkDataBufferFree(&line_coords);
  MtkDataBufferFree(&sample_coords);

  printf("Failed: status code = %d\n",status_code);
  
  return status_code;
}

void usage(char *func) {
  fprintf(stderr,
"\nUsage: %s [--help] <projection/map info file>\n"
"                     <GP_GMP file> <output filename>\n",
	  func);

  fprintf(stderr,
"\n"
"Reproject MISR geometric parameters to the given map projection.\n"
"Resampling is by nearest-neighbor.\n"
"\n"
"Result is written to an HDF-EOS file in the following fields:\n"
"    SolarAzimuth                     Solar azimuth angle.\n"
"    SolarZenith                      Solar zenith angle.\n"
"    <cam>Azimuth                     View azimuth angle for this camera.\n"
"    <cam>Zenith                      View zenith angle for this camera.\n"
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
	  );

  fprintf(stderr,
"COMMAND-LINE ARGUMENTS\n"
"\n"
"<projection/map info file>\n"
"    Text format file map projection for the output file.  Parameters are\n"
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
"         UL - Upper Left (min X, max Y)\n"
"         UR - Upper Right (max X, max Y)\n"
"         LL - Lower Left (min X, min Y)\n"
"         LR - Lower Right (max X, min Y)\n"
"\n"
"       Possible values are:\n"
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
"       origin_code = UL        # Min x, Max y; Line=Y, Sample=X\n"
"       pix_reg_code = CENTER\n"
"\n"
"\n"
"<GP_GMP file>\n"
"    MISR Geometric parameters (GP_GMP).\n"
"\n"
"<output filename>\n"
"    Filename for output file.\n"
"\n"
"Example:\n"
"\n"
"     MtkResampleGeom proj_map_info.txt \\\n"
"         MISR_AM1_GP_GMP_P011_O013842_F03_0013.hdf \\\n"
"         GEOM_O013824_maine.hdf\n"
"\n"
	  );
}

int process_args(int argc, char *argv[], argr_type *argr) {

  MTKt_status status_code = MTK_FAILURE;
  extern char *optarg;
  extern int optind;
  int ch;

  /* options descriptor */
  static struct option longopts[] = {
    { "help", no_argument,       0, 'h' },
    { 0,      0,                 0,  0 }
  };

  while ((ch = getopt_long(argc, argv, "h", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (argc-optind != 3) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  argr->proj_map_file = argv[optind++];
  argr->geom_file = argv[optind++];
  argr->output_filename = argv[optind++];

  return MTK_SUCCESS;
 ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}

