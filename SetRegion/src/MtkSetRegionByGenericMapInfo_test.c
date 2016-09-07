/*===========================================================================
=                                                                           =
=                       MtkSetRegionByGenericMapInfo_test                   =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrSetRegion.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <proj.h> 		/* Definition of MAXPROJ */

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 100 * fabs(x))

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_GenericMapInfo map_info = MTKT_GENERICMAPINFO_INIT;
  MTKt_GCTPProjInfo proj_info = MTKT_GCTPPROJINFO_INIT;
  MTKt_Region region = MTKT_REGION_INIT; 
  int path = 11;
  double min_x = 1930612.449614;  /* Albers min X coordinate */
  double min_y = 2493633.488881;  /* Albers min Y coordinate */
  double resolution_x = 250.0;    /* Albers map resolution */
  double resolution_y = 250.0;    /* Albers map resolution */
  int number_pixel_x = 1311;      /* Albers map size */
  int number_pixel_y = 2078;      /* Albers map size */
  int origin_code = MTKe_ORIGIN_UL;  
  int pix_reg_code = MTKe_PIX_REG_CENTER;  
  int proj_code = 3;		/* Albers */
  int sphere_code = 8;		/* GRS80 ellipsoid */
  int zone_code = -1;
  double proj_param[15] = { 0.0, 0.0, 29030000.0, 45030000.0, -96000000.0, 23000000.0, 
			    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  
  MTK_PRINT_STATUS(cn,"Testing MtkSetRegionByGenericMapInfo");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfo(1)\n");
    error = MTK_TRUE;
  }

  status = MtkGCTPProjInfo(proj_code,
			   sphere_code,
			   zone_code,
			   proj_param,
			   &proj_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPProjInfo(1)\n");
    error = MTK_TRUE;
  }


  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByGenericMapInfo(&map_info,
					&proj_info,
					path,
					&region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByGenericMapInfo(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  { 
    double lat_expect = 45.086198549442137562;
    double lon_expect = -69.023452978309492778;
    double xlat_expect = 279022.12649387773126;
    double ylon_expect = 196943.02607533431728;
    
    if (MTKm_CMP_NE_DBL(region.geo.ctr.lat, lat_expect) ||
	MTKm_CMP_NE_DBL(region.geo.ctr.lon, lon_expect) ||
	MTKm_CMP_NE_DBL(region.hextent.xlat, xlat_expect) ||
	MTKm_CMP_NE_DBL(region.hextent.ylon, ylon_expect)) {
      fprintf(stderr,"region.geo.ctr.lat = %20.20g (expected %20.20g)\n",
	      region.geo.ctr.lat, lat_expect);
      fprintf(stderr,"region.geo.ctr.lon = %20.20g (expected %20.20g)\n",
	      region.geo.ctr.lon, lon_expect);
      fprintf(stderr,"region.hextent.lat = %20.20g (expected %20.20g)\n",
	      region.hextent.xlat, xlat_expect);
      fprintf(stderr,"region.hextent.lon = %20.20g (expected %20.20g)\n",
	      region.hextent.ylon, ylon_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByGenericMapInfo(NULL,
					&proj_info,
					path,
					&region);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /*                 Proj_info->proj_code > MAXPROJ                     */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByGenericMapInfo(&map_info,
					NULL,
					path,
					&region);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }


  proj_info.proj_code = MAXPROJ+1;
  status = MtkSetRegionByGenericMapInfo(&map_info,
					&proj_info,
					path,
					&region);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  proj_info.proj_code = MAXPROJ;

  /* ------------------------------------------------------------------ */
  /* Argument check: Region == NULL                                     */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByGenericMapInfo(&map_info,
					&proj_info,
					path,
					NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Report test result.                                                */
  /* ------------------------------------------------------------------ */
      
  if (error) {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  } else {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  }
}
