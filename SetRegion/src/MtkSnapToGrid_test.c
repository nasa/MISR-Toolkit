/*===========================================================================
=                                                                           =
=                            MtkSnapToGrid_test                             =
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
#include "MisrMapQuery.h"
#include "math.h"
#include <float.h>
#include <stdio.h>

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 100 * fabs(x))

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  MTKt_Region region;		/* Region structure */
  MTKt_MapInfo mapinfo;		/* Map Info structure */
  int path;			/* Path */
  int resolution;		/* Resolution */
  char extent_units[10];	/* Extent units */
  double ctr_lat_dd;		/* Center latitude in decimal degrees */
  double ctr_lon_dd;		/* Center longitude in decimal degrees */
  double lat_extent;		/* Latitude extent */
  double lon_extent;		/* Longitude extent */
  int sblock;			/* Start block */
  int eblock;			/* End block */
  int cn = 0;			/* Column number */

  MTK_PRINT_STATUS(cn,"Testing MtkSnapToGrid");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  path = 37;
  resolution = 1100;
  sblock = 1;
  eblock = 1;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByPathBlockRange(1)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 128;
    int nsample_expect = 512;
    int resfactor_expect = 4;
    int resolution_expect = 1100;
    double som_ulc_x_expect = 7461300;
    double som_ulc_y_expect = 528000;
    double som_lrc_x_expect = 7601000;
    double som_lrc_y_expect = 1090100;
    double som_ctr_x_expect = 7531150;
    double som_ctr_y_expect = 809050;
    int pp_nline_expect = 128;
    int pp_nsample_expect = 512;
    int pp_resolution_expect = 1100;
    double ctr_lat_expect = 66.11943664383446162;
    double ctr_lon_expect = 48.412850619515594985;
    double ulc_lat_expect = 66.22632060370290219;
    double ulc_lon_expect = 54.829919817583743225;
    double lrc_lat_expect = 65.750320116186372843;
    double lrc_lon_expect = 42.11468173345903665;
    double llc_lat_expect = 67.442208178545470787;
    double llc_lon_expect = 54.077023422045421341;
    double urc_lat_expect = 64.610517766695764408;
    double urc_lon_expect = 43.346176776542208131;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect) ||
	mapinfo.pp.nline != pp_nline_expect ||
	mapinfo.pp.nsample != pp_nsample_expect ||
	mapinfo.pp.resolution != pp_resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lat,ctr_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lon,ctr_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lat,ulc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lon,ulc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lat,lrc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lon,lrc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lat,llc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lon,llc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lat,urc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lon,urc_lon_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"pp.nline = %d (expected %d)\n",
	      mapinfo.pp.nline, pp_nline_expect);
      fprintf(stderr,"pp.nsample = %d (expected %d)\n",
	      mapinfo.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"pp.resolution = %d (expected %d)\n",
	      mapinfo.pp.resolution, pp_resolution_expect);
      fprintf(stderr,"geo.ctr.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lat, ctr_lat_expect);
      fprintf(stderr,"geo.ctr.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lon, ctr_lon_expect);
      fprintf(stderr,"geo.ulc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lat, ulc_lat_expect);
      fprintf(stderr,"geo.ulc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lon, ulc_lon_expect);
      fprintf(stderr,"geo.lrc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lat, lrc_lat_expect);
      fprintf(stderr,"geo.lrc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lon, lrc_lon_expect);
      fprintf(stderr,"geo.llc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lat, llc_lat_expect);
      fprintf(stderr,"geo.llc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lon, llc_lon_expect);
      fprintf(stderr,"geo.urc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lat, urc_lat_expect);
      fprintf(stderr,"geo.urc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lon, urc_lon_expect);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  path = 37;
  resolution = 275;
  sblock = 1;
  eblock = 1;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByPathBlockRange(2)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(2)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 512;
    int nsample_expect = 2048;
    int resfactor_expect = 1;
    int resolution_expect = 275;
    double som_ulc_x_expect = 7461300 - 412.5;
    double som_ulc_y_expect = 528000 - 412.5;
    double som_lrc_x_expect = 7601000 + 412.5;
    double som_lrc_y_expect = 1090100 + 412.5;
    double som_ctr_x_expect = 7531150;
    double som_ctr_y_expect = 809050;
    int pp_nline_expect = 512;
    int pp_nsample_expect = 2048;
    int pp_resolution_expect = 275;
    double ctr_lat_expect = 66.11943664383446162;
    double ctr_lon_expect = 48.412850619515594985;
    double ulc_lat_expect = 66.223580786245165086;
    double ulc_lon_expect = 54.840947573401336967;
    double lrc_lat_expect = 65.752107018876998268;
    double lrc_lon_expect = 42.102734935216133749;
    double llc_lat_expect = 67.446688516795447299;
    double llc_lon_expect = 54.084028800712005136;
    double urc_lat_expect = 64.605643434536489167;
    double urc_lon_expect = 43.341781178491309845;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect) ||
	mapinfo.pp.nline != pp_nline_expect ||
	mapinfo.pp.nsample != pp_nsample_expect ||
	mapinfo.pp.resolution != pp_resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lat,ctr_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lon,ctr_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lat,ulc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lon,ulc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lat,lrc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lon,lrc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lat,llc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lon,llc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lat,urc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lon,urc_lon_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"pp.nline = %d (expected %d)\n",
	      mapinfo.pp.nline, pp_nline_expect);
      fprintf(stderr,"pp.nsample = %d (expected %d)\n",
	      mapinfo.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"pp.resolution = %d (expected %d)\n",
	      mapinfo.pp.resolution, pp_resolution_expect);
      fprintf(stderr,"geo.ctr.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lat, ctr_lat_expect);
      fprintf(stderr,"geo.ctr.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lon, ctr_lon_expect);
      fprintf(stderr,"geo.ulc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lat, ulc_lat_expect);
      fprintf(stderr,"geo.ulc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lon, ulc_lon_expect);
      fprintf(stderr,"geo.lrc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lat, lrc_lat_expect);
      fprintf(stderr,"geo.lrc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lon, lrc_lon_expect);
      fprintf(stderr,"geo.llc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lat, llc_lat_expect);
      fprintf(stderr,"geo.llc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lon, llc_lon_expect);
      fprintf(stderr,"geo.urc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lat, urc_lat_expect);
      fprintf(stderr,"geo.urc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lon, urc_lon_expect);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 3                                                      */
  /* ------------------------------------------------------------------ */

  path = 39;
  resolution = 1100;
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 10;
  lon_extent = 33;
  sprintf(extent_units, "%dm", resolution);

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      extent_units,
				      &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble MtkSetRegionByLatLonExtent(3)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(3)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 32;
    int nsample_expect = 48;
    int resfactor_expect = 4;
    int resolution_expect = 1100;
    double som_ulc_x_expect = 16173300;
    double som_ulc_y_expect = 492800;
    double som_lrc_x_expect = 16207400;
    double som_lrc_y_expect = 544500;
    double som_ctr_x_expect = 16190350;
    double som_ctr_y_expect = 518650;
    int pp_nline_expect = 128;
    int pp_nsample_expect = 512;
    int pp_resolution_expect = 1100;
    double ctr_lat_expect = 35.035883301307627846 ;
    double ctr_lon_expect = -114.92522088534047953 ;
    double ulc_lat_expect = 35.216449778288378525 ;
    double ulc_lon_expect = -115.1842952441491974 ;
    double lrc_lat_expect = 34.85476656498944692 ;
    double lrc_lon_expect = -114.66728660037304621 ;
    double llc_lat_expect = 34.911252742717501008 ;
    double llc_lon_expect = -115.22864920331554117 ;
    double urc_lat_expect = 35.159750523726522431 ;
    double urc_lon_expect = -114.62087379924381025 ;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect) ||
	mapinfo.pp.nline != pp_nline_expect ||
	mapinfo.pp.nsample != pp_nsample_expect ||
	mapinfo.pp.resolution != pp_resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lat,ctr_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ctr.lon,ctr_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lat,ulc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.ulc.lon,ulc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lat,lrc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.lrc.lon,lrc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lat,llc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.llc.lon,llc_lon_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lat,urc_lat_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.geo.urc.lon,urc_lon_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"pp.nline = %d (expected %d)\n",
	      mapinfo.pp.nline, pp_nline_expect);
      fprintf(stderr,"pp.nsample = %d (expected %d)\n",
	      mapinfo.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"pp.resolution = %d (expected %d)\n",
	      mapinfo.pp.resolution, pp_resolution_expect);
      fprintf(stderr,"geo.ctr.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lat, ctr_lat_expect);
      fprintf(stderr,"geo.ctr.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ctr.lon, ctr_lon_expect);
      fprintf(stderr,"geo.ulc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lat, ulc_lat_expect);
      fprintf(stderr,"geo.ulc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.ulc.lon, ulc_lon_expect);
      fprintf(stderr,"geo.lrc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lat, lrc_lat_expect);
      fprintf(stderr,"geo.lrc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.lrc.lon, lrc_lon_expect);
      fprintf(stderr,"geo.llc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lat, llc_lat_expect);
      fprintf(stderr,"geo.llc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.llc.lon, llc_lon_expect);
      fprintf(stderr,"geo.urc.lat = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lat, urc_lat_expect);
      fprintf(stderr,"geo.urc.lon = %20.20g (expected %20.20g)\n",
	      mapinfo.geo.urc.lon, urc_lon_expect);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 4                                                      */
  /* ------------------------------------------------------------------ */

  path = 37;
  resolution = 1100;
  sblock = 89;
  eblock = 180;

  status = MtkSetRegionByPathBlockRange(path, sblock, eblock, &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble MtkSetRegionByPathBlockRange(4)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(4)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 11776;
    int nsample_expect = 1328;
    int resfactor_expect = 4;
    int resolution_expect = 1100;
    double som_ulc_x_expect = 19851700;
    double som_ulc_y_expect = -1144000;
    double som_lrc_x_expect = 32804200;
    double som_lrc_y_expect =   315700;
    double som_ctr_x_expect = 26327950;
    double som_ctr_y_expect =  -414150;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"Unexpected result(test4).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 5                                                      */
  /* ------------------------------------------------------------------ */

  path = 39;
  resolution = 8800;
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100000.0;
  lon_extent = 1100000.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble MtkSetRegionByLatLonExtent(5)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(5)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 126;
    int nsample_expect = 128;
    int resfactor_expect = 32;
    int resolution_expect = 8800;
    double som_ulc_x_expect = 15649150;
    double som_ulc_y_expect =    -48950 ;
    double som_lrc_x_expect =  16749150 ;
    double som_lrc_y_expect =   1068650 ;
    double som_ctr_x_expect =  16199150 ;
    double som_ctr_y_expect =    509850 ;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"Unexpected result(test5).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 6                                                      */
  /* ------------------------------------------------------------------ */

  path = 37;
  resolution = 275;
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100000.0;
  lon_extent = 1100000.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble MtkSetRegionByLatLonExtent(6)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(6)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 4096;
    int nsample_expect = 4096;
    int resfactor_expect = 1;
    int resolution_expect = 275;
    double som_ulc_x_expect =15662487.5;
    double som_ulc_y_expect = -334812.5;
    double som_lrc_x_expect = 16788612.5;
    double som_lrc_y_expect =  791312.5;
    double som_ctr_x_expect =  16225550;
    double som_ctr_y_expect =    228250;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"Unexpected result(test6).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 7                                                      */
  /* ------------------------------------------------------------------ */

  path = 37;
  resolution = 275;
  ctr_lat_dd = 35.0;
  ctr_lon_dd = -115.0;
  lat_extent = 1100001.0;
  lon_extent = 1099999.0;

  status = MtkSetRegionByLatLonExtent(ctr_lat_dd, ctr_lon_dd,
				      lat_extent,
				      lon_extent,
				      "meters",
				      &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble MtkSetRegionByLatLonExtent(7)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution, region, &mapinfo);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(7)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
 
  {
    int nline_expect = 4096;
    int nsample_expect = 4096;
    int resfactor_expect = 1;
    int resolution_expect = 275;
    double som_ulc_x_expect =15662487.5;
    double som_ulc_y_expect = -334812.5;
    double som_lrc_x_expect = 16788612.5;
    double som_lrc_y_expect =  791312.5;
    double som_ctr_x_expect =  16225550;
    double som_ctr_y_expect =    228250;

    if (mapinfo.nline != nline_expect ||
	mapinfo.nsample != nsample_expect ||
	mapinfo.resfactor != resfactor_expect ||
	mapinfo.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.lrc.y, som_lrc_y_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.x, som_ctr_x_expect) ||
	MTKm_CMP_NE_DBL(mapinfo.som.ctr.y, som_ctr_y_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      mapinfo.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      mapinfo.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      mapinfo.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      mapinfo.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"som.ctr.x = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.x, som_ctr_x_expect);
      fprintf(stderr,"som.ctr.y = %20.20g (expected %20.20g)\n",
	      mapinfo.som.ctr.y, som_ctr_y_expect);
      fprintf(stderr,"Unexpected result(test7).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info_out = NULL                                */
  /* ------------------------------------------------------------------ */

  status = MtkSnapToGrid(path, resolution, region, NULL);
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
