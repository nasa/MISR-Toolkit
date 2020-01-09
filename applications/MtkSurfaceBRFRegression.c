/*===========================================================================
=                                                                           =
=                               MtkSurfaceBRFRegression                     =
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
#define LAND_BRF_FIELD_NAME_NC "Bidirectional_Reflectance_Factor"  // For land version 23 and later
#define LAND_BRF_GRID_NAME_NC "1.1_KM_PRODUCTS"                    // For land version 23 and later
#define LAND_BRF_FIELD_NAME_HDF "LandBRF"        // For land version 22 and earlier
#define LAND_BRF_GRID_NAME_HDF "SubregParamsLnd" // For land version 22 and earlier
#define AGP_GRID_NAME "Standard"
#define AGP_LAND_WATER_ID_FIELD_NAME "SurfaceFeatureID"
#define AGP_LW_LAND 1 
#define GLITTER_THRESHOLD_DEFAULT 40.0
#define MAX_NUMBER_GP_GMP_FIELD 5
#define NUMBER_BAND 4
#define NUMBER_CAMERA 9
#define OUTPUT_GRIDNAME "SurfaceBRFRegression"

/* -------------------------------------------------------------------------*/
/* Structure to contain command-line arguments.                             */
/* -------------------------------------------------------------------------*/

typedef struct {
  char *proj_map_file;  /* Projection/map info file */
  char *land_file;      /* AS_LAND file */
  char *terrain_file;   /* GRP_TERRAIN file*/
  char *output_basename;   /* Output basename*/
  char agp_file[MAX_FILENAME]; 	/* AGP file (optional) */
  char geom_file[MAX_FILENAME]; /* GP_GMP file (optional) */
  float glitter_threshold; 	/* Threshold for determining 
				   glitter contamination. */
  int  band;            /* Band to process. -1 = all */
  int output; /* 4 to output HDF4/HDF-EOS2, 5 for HDF5/HDF-EOS5 */
} argr_type;	   /* Argument parse result */

#define ARGR_TYPE_INIT {NULL, NULL, NULL, NULL, {0}, {0}, GLITTER_THRESHOLD_DEFAULT, -1, 4}

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
  MTKt_DataBuffer land_brf_data = MTKT_DATABUFFER_INIT; 
				/* LandBRF data */
  MTKt_DataBuffer land_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for LandBRF data */
  MTKt_DataBuffer land_brf_mask_upsampled = MTKT_DATABUFFER_INIT; 
				/* Valid mask for LandBRF data, upsampled to
				   full resolution of terrain data. */
  MTKt_DataBuffer land_brf_sigma = MTKT_DATABUFFER_INIT; 
				/* Land BRF sigma */
  MTKt_MapInfo    land_brf_map_info = MTKT_MAPINFO_INIT;
				/* Map information for LandBRF data  */
  MTKt_DataBuffer toa_brf_data = MTKT_DATABUFFER_INIT; 
				/* Top-of-atmosphere BRF data */
  MTKt_MapInfo    toa_rad_rdqi_map_info = MTKT_MAPINFO_INIT;
				/* Map information for terrain RDQI.  */
  MTKt_DataBuffer toa_rad_rdqi = MTKT_DATABUFFER_INIT; 
				/* Top-of-atmosphere radiance data quality indicator */ 
  MTKt_DataBuffer toa_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for TOA BRF. */
  MTKt_DataBuffer toa_brf_data_1100 = MTKT_DATABUFFER_INIT; 
				/* Top-of-atmosphere BRF data at 1100 meters per pixel. */
  MTKt_DataBuffer toa_brf_mask_1100 = MTKT_DATABUFFER_INIT; 
				/* Valid mask for 1100 meter TOA BRF. */
  MTKt_MapInfo    toa_brf_map_info = MTKT_MAPINFO_INIT;
				/* Map information for terrain-projected radiance.  */
  MTKt_RegressionCoeff 
                  regression_coeff = MTKT_REGRESSION_COEFF_INIT;
				/* Regression coefficients. */
  MTKt_DataBuffer smooth_tmp = MTKT_DATABUFFER_INIT;
				/* Temporary buffer to hold smoothed data. */
  MTKt_MapInfo    regression_coeff_map_info = MTKT_MAPINFO_INIT;
				/* Map info for regression coefficients. */
  MTKt_DataBuffer regressed_data = MTKT_DATABUFFER_INIT; 
				/* Regression result. */
  MTKt_DataBuffer regressed_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for regression result. */
  MTKt_DataBuffer latitude = MTKT_DATABUFFER_INIT; 
				/* Temporary latitude array for reprojection. */
  MTKt_DataBuffer longitude = MTKT_DATABUFFER_INIT; 
				/* Temporary longitude array for reprojection. */
  MTKt_DataBuffer line_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary line coord array for reprojection. */
  MTKt_DataBuffer sample_coords = MTKT_DATABUFFER_INIT; 
				/* Temporary sample coord array for reprojection. */
  MTKt_DataBuffer resampled_regressed_brf_data = MTKT_DATABUFFER_INIT; 
				/* Final result, resampled to target map. */
  MTKt_DataBuffer resampled_regressed_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for final result. */
  MTKt_DataBuffer resampled_land_brf_data = MTKT_DATABUFFER_INIT; 
				/* Input LandBRF, resampled to target map. */
  MTKt_DataBuffer resampled_land_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for final result. */
  MTKt_DataBuffer resampled_toa_brf_data = MTKT_DATABUFFER_INIT; 
				/* Input TOA BRF, resampled to target map. */
  MTKt_DataBuffer resampled_toa_brf_mask = MTKT_DATABUFFER_INIT; 
				/* Valid mask for final result. */
  MTKt_DataBuffer glitter_data = MTKT_DATABUFFER_INIT; 
				/* Glitter angle field. */
  MTKt_DataBuffer agp_land_water_id_data = MTKT_DATABUFFER_INIT; 
				/* AGP land/water identifier. */
  MTKt_MapInfo    agp_land_water_id_map_info = MTKT_MAPINFO_INIT;
				/* Map information for AGP land/water id field. */
  MTKt_DataBuffer glitter_mask = MTKT_DATABUFFER_INIT; 
				/* Mask indicating glitter contaminated locations. */
  MTKt_MapInfo    glitter_map_info = MTKT_MAPINFO_INIT;
				/* Map information for glitter angle field. */
  MTKt_DataBuffer resampled_glitter_data = MTKT_DATABUFFER_INIT; 
				/* Glitter mask, resampled to target map. */
  int glitter_filter = 0;	/* Flag indicating if glitter filtering should
				   be done. */
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
  float32 fill_uint8 = 0;	/* Fill value for uint8 fields. */

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

  status = MtkFileToPath(argr.land_file, &path);
  if (status != MTK_SUCCESS) {
      printf("\n\nCheck that AS_LAND filename is correct: %s\n",argr.land_file);
      MTK_ERR_MSG_JUMP("Trouble with MtkFileToPath(land)\n");
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
  
  if (argr.output == 4) {
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
  } else { /* HDF5 output rather than 4 */
    /* 
    fid = HE5_GDopen(outputfile,DFACC_CREATE);
    if (fid == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with HE5_GDopen\n");
    }
	
    hstatus = HE5_GDcreate(fid,OUTPUT_GRIDNAME,size_x, size_y, 
  		     upleft, lowright);
  
    gid = HE5_GDattach(fid, OUTPUT_GRIDNAME);
    if (gid == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with HE5_GDattach\n");
    }
  
    hstatus = HE5_GDdeforigin(gid,target_map_info.origin_code);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with HE5_GDdeforigin\n");
    }
  
    hstatus = HE5_GDdefpixreg(gid,target_map_info.pix_reg_code);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with HE5_GDdefpixreg\n");
    }

    hstatus = HE5_GDdefproj(gid,target_proj_info.proj_code,
  		      target_proj_info.zone_code, target_proj_info.sphere_code,
  		      target_proj_info.proj_param);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with HE5_GDdefproj\n");
    }  */
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
  /* If AGP and geometric parameters are provided...                    */
  /* ------------------------------------------------------------------ */

  if (argr.agp_file[0] && argr.geom_file[0]) {
    glitter_filter = 1;

    printf("Calculating glitter mask...\n");

  /* ------------------------------------------------------------------ */
  /* Read glitter angle for the target area.                            */
  /* ------------------------------------------------------------------ */

    snprintf(fieldname,MAX_FIELDNAME,"%sGlitter",camera_name[camera]);
    status = MtkReadData(argr.geom_file, GP_GMP_GRID_NAME, fieldname,
			 region, &glitter_data, &glitter_map_info);
    if (status != MTK_SUCCESS) {
      printf("\n\nCheck that GP_GMP filename is correct: %s\n",argr.geom_file);
      MTK_ERR_MSG_JUMP("Trouble with MtkReadData(geom)\n");
    }

  /* ------------------------------------------------------------------ */
  /* Read AGP land/water identifier for the target area.                */
  /* ------------------------------------------------------------------ */

    status = MtkReadData(argr.agp_file, AGP_GRID_NAME, 
			 AGP_LAND_WATER_ID_FIELD_NAME,
			 region, &agp_land_water_id_data, 
			 &agp_land_water_id_map_info);
    if (status != MTK_SUCCESS) {
      printf("\n\nCheck that AGP filename is correct: %s\n",argr.agp_file);
      MTK_ERR_MSG_JUMP("Trouble with MtkReadData(agp)\n");
    }

  /* ------------------------------------------------------------------ */
  /* Generate glitter mask at 1.1 km.                                   */
  /* Glitter mask is set to 1 where location is glitter contaminated.   */
  /* Otherwise set to 0.                                                */
  /* ------------------------------------------------------------------ */

    status = MtkDataBufferAllocate(agp_land_water_id_data.nline, 
				   agp_land_water_id_data.nsample,
				   MTKe_uint8, &glitter_mask);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkDataBufferAllocate(glitter_mask)\n");
    }
    
    for (iline = 0; iline < agp_land_water_id_data.nline; iline++) {
      for (isample = 0; isample < agp_land_water_id_data.nsample; isample++) {
	if (agp_land_water_id_data.data.u8[iline][isample] != AGP_LW_LAND &&
	    glitter_data.data.d[iline/16][isample/16] > 0 &&
	    glitter_data.data.d[iline/16][isample/16] < argr.glitter_threshold) {
	  glitter_mask.data.u8[iline][isample] = 1;
	} else {
	  glitter_mask.data.u8[iline][isample] = 0;
	}
      }
    }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

    MtkDataBufferFree(&agp_land_water_id_data);
    MtkDataBufferFree(&glitter_data);

  /* ------------------------------------------------------------------ */
  /* Reproject data to the target map.                                  */ 
  /* ------------------------------------------------------------------ */

    status = MtkTransformCoordinates(agp_land_water_id_map_info,
				     latitude, 
				     longitude,
				     &line_coords,
				     &sample_coords);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkTransformCoordinates\n");
    }

    status = MtkResampleNearestNeighbor(glitter_mask, 
					line_coords, 
					sample_coords, 
					&resampled_glitter_data);
    if (status != MTK_SUCCESS) {
      MTK_ERR_MSG_JUMP("Trouble with MtkResampleCubicConvolution\n");
    }

  /* ------------------------------------------------------------------ */
  /* Write reprojected glitter mask to HDF-EOS file.                    */
  /* ------------------------------------------------------------------ */
      
    snprintf(fieldname,MAX_FIELDNAME,"Glitter_Mask_%s",camera_name[camera]);
    hstatus = GDdeffield(gid, fieldname, dimlist, DFNT_UINT8, 0);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDdeffield(glitter)\n");
    }
    
    hstatus = GDsetfillvalue(gid, fieldname, &fill_uint8);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDsetfillvalue()\n");
    }
    
    hstatus = GDsettilecomp(gid, fieldname, tile_rank, 
			    tile_dims, comp_code, comp_parm);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDsettilecomp()\n");
    }

    hstatus = GDwritefield(gid, fieldname, NULL, NULL, edge, 
			   resampled_glitter_data.dataptr);
    if (hstatus == FAIL) {
      MTK_ERR_MSG_JUMP("Trouble with GDwritefield(glitter)\n");
    }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&line_coords);
      MtkDataBufferFree(&sample_coords);
      MtkDataBufferFree(&resampled_glitter_data);

  /* ------------------------------------------------------------------ */
  /* End if AGP and geometric parameters are provided.                  */
  /* ------------------------------------------------------------------ */

  }

  /* ------------------------------------------------------------------ */
  /* Detect land file type                                              */
  /* ------------------------------------------------------------------ */

  int is_netcdf = 0;
  {
    int ncid;
    int status = nc_open(argr.land_file, NC_NOWRITE, &ncid);
    if (status == NC_NOERR) {
      is_netcdf = 1;
    }
  }

  /* ------------------------------------------------------------------ */
  /* For each band to process...                                        */
  /* ------------------------------------------------------------------ */

  for (iband = 0 ; iband < NUMBER_BAND ; iband++)  {
    if (argr.band < 0 || iband == argr.band) {

      printf("Processing band %d (%s)...\n",iband, band_name[iband]);

  /* ------------------------------------------------------------------ */
  /* Read LandBRF for the target area.                                  */
  /* ------------------------------------------------------------------ */

      if (is_netcdf) {
        snprintf(fieldname,MAX_FIELDNAME,"%s[%d][%d]",LAND_BRF_FIELD_NAME_NC,iband,camera);
        status = MtkReadData(argr.land_file, LAND_BRF_GRID_NAME_NC, fieldname,
                             region, &land_brf_data, &land_brf_map_info);  // Try netCDF
      } else {
        snprintf(fieldname,MAX_FIELDNAME,"%s[%d][%d]",LAND_BRF_FIELD_NAME_HDF,iband,camera);
				status = MtkReadData(argr.land_file, LAND_BRF_GRID_NAME_HDF, fieldname,
														 region, &land_brf_data, &land_brf_map_info);  // Try HDF
			}
      if (status != MTK_SUCCESS) {
				MTK_ERR_MSG_JUMP("Trouble with MtkReadData(LandBRF)\n");
      }

  /* ------------------------------------------------------------------ */
  /* Generate mask indicating valid LandBRF.                            */
  /* ------------------------------------------------------------------ */

      status = MtkDataBufferAllocate(land_brf_data.nline, land_brf_data.nsample,
				     MTKe_uint8,&land_brf_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkDataBufferAllocate(toa_brf_mask)\n");
      }

      for (iline = 0; iline < land_brf_data.nline; iline++) {
        for (isample = 0; isample < land_brf_data.nsample; isample++) {
          if (0 == is_netcdf) {
            if (land_brf_data.data.f[iline][isample] > 65532.0 ||
                land_brf_data.data.f[iline][isample] == 0.0) {
              land_brf_mask.data.u8[iline][isample] = 0;
            } else {
              land_brf_mask.data.u8[iline][isample] = 1;
            }
          } else {
            if (land_brf_data.data.f[iline][isample] == -9999 ||
                land_brf_data.data.f[iline][isample] == 0.0) {
              land_brf_mask.data.u8[iline][isample] = 0;
            } else {
              land_brf_mask.data.u8[iline][isample] = 1;
            }
          }
        }
      }

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
  /* Generate Land BRF sigma.                                           */
  /* Set all to 1.0.  This gives equal weight to all LandBRF values.    */
  /* ------------------------------------------------------------------ */

      status = MtkDataBufferAllocate(land_brf_data.nline, land_brf_data.nsample,
				     MTKe_float,&land_brf_sigma);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkDataBufferAllocate(land_brf_sigma)\n");
      }

      for (iline = 0; iline < land_brf_data.nline; iline++) {
	for (isample = 0; isample < land_brf_data.nsample; isample++) {
	  land_brf_sigma.data.f[iline][isample] = 1.0;
	}
      }

  /* ------------------------------------------------------------------ */
  /* Downsample terrain data to resolution of LandBRF.                  */
  /* ------------------------------------------------------------------ */

      status = MtkDownsample(&toa_brf_data,
			     &toa_brf_mask,
			     (land_brf_map_info.resolution / 
			      toa_brf_map_info.resolution),
			     &toa_brf_data_1100,
			     &toa_brf_mask_1100);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkDownsample\n");
      }

  /* ------------------------------------------------------------------ */
  /* If glitter information is available, filter out locations which    */
  /* are glitter contaminated.                                          */
  /* ------------------------------------------------------------------ */

      if (glitter_filter) {
	for (iline = 0; iline < toa_brf_data_1100.nline; iline++) {
	  for (isample = 0; isample < toa_brf_data_1100.nsample; isample++) {
	    if (glitter_mask.data.u8[iline][isample]) {
	      toa_brf_mask_1100.data.u8[iline][isample] = 0;
	    } else {
	      toa_brf_mask_1100.data.u8[iline][isample] = 1;
	    }
	  }
	}
      }

  /* ------------------------------------------------------------------ */
  /* Generate regression coefficients, for mapping TOA BRF to LandBRF.  */
  /* ------------------------------------------------------------------ */

      status = MtkRegressionCoeffCalc(&toa_brf_data_1100, 
				      &toa_brf_mask_1100, 
				      &land_brf_data, 
				      &land_brf_sigma,
				      &land_brf_mask,
				      &land_brf_map_info,
				      16, 
				      &regression_coeff,
				      &regression_coeff_map_info);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkRegressionCoeffCalc()\n");
      }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&land_brf_sigma);
      MtkDataBufferFree(&toa_brf_data_1100);
      MtkDataBufferFree(&toa_brf_mask_1100);

  /* ------------------------------------------------------------------ */
  /* Smooth regression coefficients.                                    */
  /* ------------------------------------------------------------------ */

      status = MtkSmoothData(&regression_coeff.slope, 
			     &regression_coeff.valid_mask,
			     3, 3,
			     &smooth_tmp);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkSmoothData(slope)\n");
      }

      for (iline = 0; iline < regression_coeff.valid_mask.nline; iline++) {
	for (isample = 0; isample < regression_coeff.valid_mask.nsample; isample++) {
	  if (regression_coeff.valid_mask.data.u8[iline][isample]) {
	    regression_coeff.slope.data.f[iline][isample] = 
	      smooth_tmp.data.f[iline][isample];
	  }
	}
      }

      MtkDataBufferFree(&smooth_tmp);

      status = MtkSmoothData(&regression_coeff.intercept, 
			     &regression_coeff.valid_mask,
			     3, 3,
			     &smooth_tmp);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkSmoothData(intercept)\n");
      }

      for (iline = 0; iline < regression_coeff.valid_mask.nline; iline++) {
	for (isample = 0; isample < regression_coeff.valid_mask.nsample; isample++) {
	  if (regression_coeff.valid_mask.data.u8[iline][isample]) {
	    regression_coeff.intercept.data.f[iline][isample] = 
	      smooth_tmp.data.f[iline][isample];
	  }
	}
      }

      MtkDataBufferFree(&smooth_tmp);

  /* ------------------------------------------------------------------ */
  /* Apply regression to full resolution TOA BRF.                       */
  /* ------------------------------------------------------------------ */

      status = MtkApplyRegression(&toa_brf_data,
				  &toa_brf_mask, 
				  &toa_brf_map_info, 
				  &regression_coeff,
				  &regression_coeff_map_info,
				  &regressed_data,
				  &regressed_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkApplyRegression\n");
      }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

      MtkRegressionCoeffFree(&regression_coeff);

  /* ------------------------------------------------------------------ */
  /* Mask out locations where LandBRF is not available.                 */
  /* ------------------------------------------------------------------ */

      status = MtkUpsampleMask(&land_brf_mask,
			       (land_brf_map_info.resolution / 
				toa_brf_map_info.resolution),
			       &land_brf_mask_upsampled);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkUpsampleMask\n");
      }

      for (iline = 0; iline < regressed_mask.nline; iline++) {
	for (isample = 0; isample < regressed_mask.nsample; isample++) {
	  if (land_brf_mask_upsampled.data.u8[iline][isample] == 0) {
	    regressed_mask.data.u8[iline][isample] = 0;
	  }
	}
      }

      MtkDataBufferFree(&land_brf_mask_upsampled);

  /* ------------------------------------------------------------------ */
  /* Reproject LandBRF to the target map.                               */ 
  /* ------------------------------------------------------------------ */

      status = MtkTransformCoordinates(land_brf_map_info, 
				       latitude, 
				       longitude,
				       &line_coords,
				       &sample_coords);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkTransformCoordinates\n");
      }

      status = MtkResampleCubicConvolution(&land_brf_data, 
					   &land_brf_mask, 
					   &line_coords, 
					   &sample_coords, 
					   -0.5,
					   &resampled_land_brf_data,
					   &resampled_land_brf_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkResampleCubicConvolution\n");
      }

  /* ------------------------------------------------------------------ */
  /* Free memory                                                        */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&resampled_land_brf_mask);
      MtkDataBufferFree(&land_brf_data);
      MtkDataBufferFree(&land_brf_mask);
      MtkDataBufferFree(&line_coords);
      MtkDataBufferFree(&sample_coords);

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
  /* Reproject regressed surface BRF values to target map.              */
  /* The regressed data is in the same grid as the TOA BRF, so it uses  */
  /* the same line/sample coordinates as above.                         */
  /* ------------------------------------------------------------------ */

      status = MtkResampleCubicConvolution(&regressed_data, 
					   &regressed_mask, 
					   &line_coords, 
					   &sample_coords, 
					   -0.5,
					   &resampled_regressed_brf_data,
					   &resampled_regressed_brf_mask);
      if (status != MTK_SUCCESS) {
	MTK_ERR_MSG_JUMP("Trouble with MtkResampleCubicConvolution\n");
      }


  /* ------------------------------------------------------------------ */
  /* Free memory                                                        */
  /* ------------------------------------------------------------------ */

      MtkDataBufferFree(&regressed_data);
      MtkDataBufferFree(&regressed_mask);

      MtkDataBufferFree(&line_coords);
      MtkDataBufferFree(&sample_coords);

  /* ------------------------------------------------------------------ */
  /* Write regressed BRF data to HDF-EOS file.                          */
  /* ------------------------------------------------------------------ */
      
      snprintf(fieldname,MAX_FIELDNAME,"Regressed_BRF_%s_%s",
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
			     resampled_regressed_brf_data.dataptr);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDwritefield\n");
      }

  /* ------------------------------------------------------------------ */
  /* Write regressed BRF mask to HDF-EOS file.                          */
  /* ------------------------------------------------------------------ */

      snprintf(fieldname,MAX_FIELDNAME,"Regressed_BRF_Mask_%s_%s",
	       camera_name[camera], band_name[iband]);

	
      hstatus = GDdeffield(gid, fieldname, dimlist, DFNT_UINT8, 0);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDdeffield\n");
      }

      hstatus = GDsetfillvalue(gid, fieldname, &fill_uint8);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDsetfillvalue()\n");
      }
      
      hstatus = GDsettilecomp(gid, fieldname, tile_rank, 
			      tile_dims, comp_code, comp_parm);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDsettilecomp()\n");
      }

      hstatus = GDwritefield(gid, fieldname, NULL, NULL, edge, 
			     resampled_regressed_brf_mask.dataptr);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDwritefield\n");
      }

  /* ------------------------------------------------------------------ */
  /* Write input LandBRF to HDF-EOS file.                               */
  /* ------------------------------------------------------------------ */

      snprintf(fieldname,MAX_FIELDNAME,"LandBRF_%s_%s",
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
			     resampled_land_brf_data.dataptr);
      if (hstatus == FAIL) {
	MTK_ERR_MSG_JUMP("Trouble with GDwritefield\n");
      }

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

      MtkDataBufferFree(&resampled_regressed_brf_data);
      MtkDataBufferFree(&resampled_regressed_brf_mask);
      MtkDataBufferFree(&resampled_land_brf_data);
      MtkDataBufferFree(&resampled_toa_brf_data);

  /* ------------------------------------------------------------------ */
  /* End loop for each band to process.                                 */
  /* ------------------------------------------------------------------ */

    }
  }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */
  
  if (glitter_filter) {
    MtkDataBufferFree(&glitter_mask);
  }
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
  
  MtkDataBufferFree(&glitter_data);
  MtkDataBufferFree(&glitter_mask);
  MtkDataBufferFree(&agp_land_water_id_data);
  MtkDataBufferFree(&resampled_glitter_data);
  MtkDataBufferFree(&resampled_toa_brf_mask);
  MtkDataBufferFree(&resampled_land_brf_mask);
  MtkDataBufferFree(&resampled_toa_brf_data);
  MtkDataBufferFree(&resampled_land_brf_data);
  MtkDataBufferFree(&resampled_regressed_brf_data);
  MtkDataBufferFree(&resampled_regressed_brf_mask);
  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);
  MtkDataBufferFree(&regressed_data);
  MtkDataBufferFree(&regressed_mask);
  MtkDataBufferFree(&smooth_tmp);
  MtkRegressionCoeffFree(&regression_coeff);
  MtkDataBufferFree(&toa_brf_data_1100);
  MtkDataBufferFree(&toa_brf_mask_1100);
  MtkDataBufferFree(&toa_brf_mask);
  MtkDataBufferFree(&toa_rad_rdqi);
  MtkDataBufferFree(&toa_brf_data);
  MtkDataBufferFree(&land_brf_sigma);
  MtkDataBufferFree(&land_brf_mask_upsampled);
  MtkDataBufferFree(&land_brf_mask);
  MtkDataBufferFree(&land_brf_data);
  MtkDataBufferFree(&line_coords);
  MtkDataBufferFree(&sample_coords);

  printf("Failed: status code = %d\n",status_code);
  
  return status_code;
}

void usage(char *func) {
  fprintf(stderr,
"\nUsage: %s [--help] [--band=<band number>] [--glitter-threshold=<value>]\n"
"                     [--agp=<AGP filename> --geom=<GP_GMP filename>]\n"
"                     <projection/map info file> <AS_LAND file> <GRP_TERRAIN file>\n"
"                     <output basename>\n",
	  func);

  fprintf(stderr,
"\n"
"Perform Top-of-atmosphere BRF to LandBRF regression and reproject the result to the\n"
"given map projection.\n"
"\n"
"Result is written to an HDF-EOS file in the following fields:\n"
"    Regressed_BRF_<cam>_<band>       Final result of the regression.\n"
"    Regressed_BRF_Mask_<cam>_<band>  Mask indicating valid data.\n"
"    TOA_BRF_<cam>_<band>             Input TOA BRF from GRP_TERRAIN file (reprojected).\n"
"    LandBRF_<cam>_<band>             Input LandBRF from AS_LAND file (reprojected).\n"
"    Glitter_Mask_<cam>               Mask indicating where sun glint is filtered.\n"
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
"--agp=<AGP filename>\n"
"     Optional parameter, specifying a MISR Ancillary Geographic Parameters\n"
"     (AGP) file.  If provided, the AGP land/water identifier is used in\n"
"     combination with the geometric parameters to filter possible sun glint\n"
"     contamination from the top-of-atmosphere BRF.  Requires --geom.\n"
"\n"
"--geom=<GP_GMP filename>\n"
"     Optional parameter, specifying a MISR Geometric Parameters file (GP_GMP).\n"
"     This is required if --agp is set.\n"
"\n"
"--glitter-threshold=<value>\n"
"     Glitter angle at which a water surface is considered to be glitter\n"
"     contaminated in the top-of-atmosphere BRF.  Default is 40.0 degrees.\n"
"     Glitter angle the angle between a vector from the observed point to\n"
"     the camera and a vector pointing in the specular reflection direction.\n"
"     Small glitter angles indicate the possibility of observing sun glint.\n"
"\n"
"--output=<type number>\n"
"     Integer dentifier for output file HDF type. Use 4 for HDF4/HDF-EOS2 or use \n"
"     5 for HDF5/HDF-EOS5. Defaults to 4.\n"
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
"<AS_LAND file>\n"
"    MISR Land Surface parameters (AS_LAND).\n"
"\n"
"<GRP_TERRAIN>\n"
"    MISR Terrain-projected Radiance Product (GRP_TERRAIN).\n"
"\n"
"<output basename>\n"
"    Basename for output file.\n"
"\n"
"Examples:\n"
"\n"
"     MtkSurfaceBRFRegression --band=2 proj_map_info.txt \\\n"
"         MISR_AM1_AS_LAND_P011_O013824_F06_0019.b52-56.hdf \\\n"
"         MISR_AM1_GRP_TERRAIN_GM_P011_O013824_DF_F03_0024.b52-56.hdf \\\n"
"         BRF_O013824_maine\n"
"\n"
"     MtkSurfaceBRFRegression \\\n"
"         --geom=MISR_AM1_GP_GMP_P011_O013824_F03_0013.hdf \\\n"
"         --agp=MISR_AM1_AGP_P011_F01_24.hdf \\\n"
"         proj_map_info.txt \\\n"
"         MISR_AM1_AS_LAND_P011_O013824_F06_0019.b52-56.hdf \\\n"
"         MISR_AM1_GRP_TERRAIN_GM_P011_O013824_DF_F03_0024.b52-56.hdf \\\n"
"         BRF_O013824_maine\n"
"\n"
"     MtkSurfaceBRFRegression --glitter-threshold=30.0\\\n"
"         --geom=MISR_AM1_GP_GMP_P011_O013824_F03_0013.hdf \\\n"
"         --agp=MISR_AM1_AGP_P011_F01_24.hdf \\\n"
"         proj_map_info.txt \\\n"
"         MISR_AM1_AS_LAND_P011_O013824_F06_0019.b52-56.hdf \\\n"
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
    { "agp", required_argument,  0, 'a' },
    { "geom", required_argument, 0, 'g'},
    { "band", required_argument, 0, 'b' },
    { "help", no_argument,       0, 'h' },
    { "glitter-threshold", required_argument, 0, 't'},
    { "output", required_argument, 0, 'o'},    
    { 0,      0,                 0,  0 }
  };

  while ((ch = getopt_long(argc, argv, "a:g:b:t:hwo:", longopts, NULL)) != -1) {

    switch(ch) {
    case 'h':
      MTK_ERR_CODE_JUMP(MTK_FAILURE);
      break;
    case 'a' : 
      strncpy(argr->agp_file,optarg,MAX_FILENAME);
      break;
    case 'g' : 
      strncpy(argr->geom_file,optarg,MAX_FILENAME);
      break;
    case 'b' : 
      argr->band = (int)atol(optarg);
      break;
    case 't' : 
      argr->glitter_threshold = atof(optarg);
      break;
    case 'o' : 
      argr->output = atoi(optarg);
      if (argr->output != 4 && argr->output != 5) {
        MTK_ERR_MSG_JUMP("Invalid arguments");  
      }
      break;      
    default:
      MTK_ERR_MSG_JUMP("Invalid arguments");
    }
  }

  if (argc-optind != 4) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  argr->proj_map_file = argv[optind++];
  argr->land_file = argv[optind++];
  argr->terrain_file = argv[optind++];
  argr->output_basename = argv[optind++];

  return MTK_SUCCESS;
 ERROR_HANDLE:
  usage(argv[0]);
  return status_code;
}

