/*===========================================================================
=                                                                           =
=                           MtkChangeMapResolution_test                     =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrMapQuery.h"
#include "MisrSetRegion.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 1000 * fabs(x))

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_MapInfo map_info_in = MTKT_MAPINFO_INIT;
  MTKt_MapInfo map_info_in2 = MTKT_MAPINFO_INIT;
  MTKt_MapInfo map_info_bad = MTKT_MAPINFO_INIT; /* Bad map info for testing argument checks. */
  MTKt_MapInfo map_info_out = MTKT_MAPINFO_INIT;
  int new_resolution = 1100;
  int resolution_in = 275;
  MTKt_Region region = MTKT_REGION_INIT;
  double center_lat = 1.0;
  double center_lon = 2.0;
  double size_x = (17600 * 20.0);
  double size_y = (17600 * 32.0);
  int path = 192;
  
  MTK_PRINT_STATUS(cn,"Testing MtkChangeMapResolution");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Setup map information.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByLatLonExtent(center_lat,center_lon,size_x,size_y,
				      "meters", &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByPathBlockRange(1)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, resolution_in, region, &map_info_in);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1  (275 to 1100 meters)                                */
  /* ------------------------------------------------------------------ */

  status = MtkChangeMapResolution(&map_info_in,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkChangeMapResolution(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    int resfactor_expect = new_resolution / MAXRESOLUTION;
    int nline_expect = map_info_in.nline / 4;
    int nsample_expect = map_info_in.nsample / 4;
    int resolution_expect = new_resolution;
    double som_ulc_x_expect = 19816500.0;
    double som_ulc_y_expect = -105600.0;
    double som_lrc_x_expect = 20185000.0;
    double som_lrc_y_expect = 474100.0;
    int pp_nline_expect = 128;
    int pp_nsample_expect = 512;
    int pp_resolution_expect = 1100;
    double ulc_lat_expect = 2.9149265913503636938;
    double ulc_lon_expect = -0.43952631283101945003;
    double lrc_lat_expect = -0.8025217572319853776;
    double lrc_lon_expect = 4.5001546289841769521;
    double llc_lat_expect = -0.40756504440666585509;
    double llc_lon_expect = -0.68880944493131168738;
    double urc_lat_expect = 2.5116843501856966903;
    double urc_lon_expect = 4.7553632747173519846;

    if (map_info_out.nline != nline_expect ||
	map_info_out.nsample != nsample_expect ||
	map_info_out.resfactor != resfactor_expect ||
	map_info_out.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(map_info_out.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.lrc.y, som_lrc_y_expect) ||
	map_info_out.pp.nline != pp_nline_expect ||
	map_info_out.pp.nsample != pp_nsample_expect ||
	map_info_out.pp.resolution != pp_resolution_expect ||
	MTKm_CMP_NE_DBL(map_info_out.geo.ulc.lat,ulc_lat_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.ulc.lon,ulc_lon_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.lrc.lat,lrc_lat_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.lrc.lon,lrc_lon_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.llc.lat,llc_lat_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.llc.lon,llc_lon_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.urc.lat,urc_lat_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.geo.urc.lon,urc_lon_expect)) {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      map_info_out.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      map_info_out.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      map_info_out.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      map_info_out.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      map_info_out.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      map_info_out.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      map_info_out.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      map_info_out.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"pp.nline = %d (expected %d)\n",
	      map_info_out.pp.nline, pp_nline_expect);
      fprintf(stderr,"pp.nsample = %d (expected %d)\n",
	      map_info_out.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"pp.resolution = %d (expected %d)\n",
	      map_info_out.pp.resolution, pp_resolution_expect);
      fprintf(stderr,"geo.ulc.lat = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.ulc.lat, ulc_lat_expect);
      fprintf(stderr,"geo.ulc.lon = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.ulc.lon, ulc_lat_expect);
      fprintf(stderr,"geo.lrc.lat = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.lrc.lat, lrc_lat_expect);
      fprintf(stderr,"geo.lrc.lon = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.lrc.lon, lrc_lat_expect);
      fprintf(stderr,"geo.llc.lat = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.llc.lat, llc_lat_expect);
      fprintf(stderr,"geo.llc.lon = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.llc.lon, llc_lat_expect);
      fprintf(stderr,"geo.urc.lat = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.urc.lat, urc_lat_expect);
      fprintf(stderr,"geo.urc.lon = %20.20g (expected %20.20g)\n",
	      map_info_out.geo.urc.lon, urc_lat_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2  (1100 to 275 meters)                                */
  /* ------------------------------------------------------------------ */

  map_info_in2 = map_info_out;
  new_resolution = 275;
  status = MtkChangeMapResolution(&map_info_in2,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkChangeMapResolution(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    int resfactor_expect = new_resolution / MAXRESOLUTION;
    int nline_expect = map_info_in2.nline * 4;
    int nsample_expect = map_info_in2.nsample * 4;
    int resolution_expect = new_resolution;
    double som_ulc_x_expect = 19816500.0 - 412.5;
    double som_ulc_y_expect = -105600.0 - 412.5;
    double som_lrc_x_expect = 20185000.0 + 412.5;
    double som_lrc_y_expect = 474100.0 + 412.5;
    int pp_nline_expect = 512;
    int pp_nsample_expect = 2048;
    int pp_resolution_expect = 275;

    if (map_info_out.nline != nline_expect ||
	map_info_out.nsample != nsample_expect ||
	map_info_out.resfactor != resfactor_expect ||
	map_info_out.resolution != resolution_expect ||
	MTKm_CMP_NE_DBL(map_info_out.som.ulc.x, som_ulc_x_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.ulc.y, som_ulc_y_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.lrc.x, som_lrc_x_expect) ||
	MTKm_CMP_NE_DBL(map_info_out.som.lrc.y, som_lrc_y_expect) ||
	map_info_out.pp.nline != pp_nline_expect ||
	map_info_out.pp.nsample != pp_nsample_expect ||
	map_info_out.pp.resolution != pp_resolution_expect)  {
      fprintf(stderr,"nline = %d (expected %d)\n",
	      map_info_out.nline, nline_expect);
      fprintf(stderr,"nsample = %d (expected %d)\n",
	      map_info_out.nsample, nsample_expect);
      fprintf(stderr,"resfactor = %d (expected %d)\n",
	      map_info_out.resfactor, resfactor_expect);
      fprintf(stderr,"resolution = %d (expected %d)\n",
	      map_info_out.resolution, resolution_expect);
      fprintf(stderr,"som.ulc.x = %20.20g (expected %20.20g)\n",
	      map_info_out.som.ulc.x, som_ulc_x_expect);
      fprintf(stderr,"som.ulc.y = %20.20g (expected %20.20g)\n",
	      map_info_out.som.ulc.y, som_ulc_y_expect);
      fprintf(stderr,"som.lrc.x = %20.20g (expected %20.20g)\n",
	      map_info_out.som.lrc.x, som_lrc_x_expect);
      fprintf(stderr,"som.lrc.y = %20.20g (expected %20.20g)\n",
	      map_info_out.som.lrc.y, som_lrc_y_expect);
      fprintf(stderr,"pp.nline = %d (expected %d)\n",
	      map_info_out.pp.nline, pp_nline_expect);
      fprintf(stderr,"pp.nsample = %d (expected %d)\n",
	      map_info_out.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"pp.resolution = %d (expected %d)\n",
	      map_info_out.pp.resolution, pp_resolution_expect);
      fprintf(stderr,"Unexpected result(test2).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info_in = NULL                                 */
  /*                 Map_info_in->nline < 1                             */
  /*                 Map_info_in->nsample < 1                           */
  /*                 Map_info_in->resolution < 1                        */
  /* ------------------------------------------------------------------ */

  status = MtkChangeMapResolution(NULL,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  map_info_bad = map_info_in;
  map_info_bad.nline = 0;
  status = MtkChangeMapResolution(&map_info_bad,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  map_info_bad = map_info_in;
  map_info_bad.nsample = 0;
  status = MtkChangeMapResolution(&map_info_bad,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  map_info_bad = map_info_in;
  map_info_bad.resolution = 0;
  status = MtkChangeMapResolution(&map_info_bad,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   New_resolution < 1                                               */
  /*   New_resolution % MAXRESOLUTION != 0                              */
  /* ------------------------------------------------------------------ */

  new_resolution = 0;
  status = MtkChangeMapResolution(&map_info_in,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  new_resolution = 1100;

  new_resolution++;
  status = MtkChangeMapResolution(&map_info_in,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  new_resolution--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info_out = NULL                                */
  /* ------------------------------------------------------------------ */

  status = MtkChangeMapResolution(&map_info_in,
				  new_resolution,
				  NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: size_x % New_resolution != 0                       */
  /*                 size_y % New_resolution != 0                       */
  /* ------------------------------------------------------------------ */

  map_info_bad = map_info_in;
  map_info_bad.nline++;
  status = MtkChangeMapResolution(&map_info_bad,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info_bad.nline--;

  map_info_bad = map_info_in;
  map_info_bad.nsample++;
  status = MtkChangeMapResolution(&map_info_bad,
				  new_resolution,
				  &map_info_out);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info_bad.nsample--;

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
