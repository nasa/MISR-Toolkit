/*===========================================================================
=                                                                           =
=                           MtkGenericMapInfo_test                          =
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
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 100 * fabs(x))

#define SIZE_TRANSFORM 4
#define RESOLUTION_X 3
#define RESOLUTION_Y 4

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_GenericMapInfo map_info = MTKT_GENERICMAPINFO_INIT;
  double min_x = 1.0;
  double min_y = 2.0;
  double resolution_x = 3.0;
  double resolution_y = 4.0;
  int number_pixel_x = 5;
  int number_pixel_y = 6;
  int origin_code = MTKe_ORIGIN_LR;   /* Line = Y ; Sample = X */
  int pix_reg_code = MTKe_PIX_REG_CENTER;
  int i;
  
  MTK_PRINT_STATUS(cn,"Testing MtkGenericMapInfo");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
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

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double max_x_expect = min_x + number_pixel_x * resolution_x;
    double max_y_expect = min_y + number_pixel_y * resolution_y;
    double tsample_expect[SIZE_TRANSFORM] = {-1, 16, -0.5, RESOLUTION_X};
    double tline_expect[SIZE_TRANSFORM] = {1, -2, -0.5, RESOLUTION_Y};
    int size_line_expect = number_pixel_y;
    int size_sample_expect = number_pixel_x;
  
    if (map_info.min_x != min_x ||
	map_info.min_y != min_y ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	map_info.resolution_x != resolution_x ||
	map_info.resolution_y != resolution_y ||
	map_info.origin_code != origin_code ||
	map_info.pix_reg_code != pix_reg_code) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < SIZE_TRANSFORM; i++) {
      if (MTKm_CMP_NE_DBL(map_info.tline[i],tline_expect[i]) ||
	  MTKm_CMP_NE_DBL(map_info.tsample[i],tsample_expect[i])) {
	fprintf(stderr,"map_info.tline[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tline[i], tline_expect[i]);
	fprintf(stderr,"map_info.tsample[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tsample[i], tsample_expect[i]);
	fprintf(stderr,"Unexpected result(test1).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  origin_code = MTKe_ORIGIN_LL; /* Line = X ; Sample = Y */
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

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double tline_expect[SIZE_TRANSFORM] = {1, -1, -0.5, RESOLUTION_X};
    double tsample_expect[SIZE_TRANSFORM] = {1, -2, -0.5, RESOLUTION_Y};
    int size_line_expect = number_pixel_x;
    int size_sample_expect = number_pixel_y;

    if (map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	map_info.origin_code != origin_code ||
	map_info.pix_reg_code != pix_reg_code) {
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < SIZE_TRANSFORM; i++) {
      if (MTKm_CMP_NE_DBL(map_info.tline[i],tline_expect[i]) ||
	  MTKm_CMP_NE_DBL(map_info.tsample[i],tsample_expect[i])) {
	fprintf(stderr,"map_info.tline[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tline[i], tline_expect[i]);
	fprintf(stderr,"map_info.tsample[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tsample[i], tsample_expect[i]);
	fprintf(stderr,"Unexpected result(test2).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 3                                                      */
  /* ------------------------------------------------------------------ */

  origin_code = MTKe_ORIGIN_UL;  /* Line = Y ; Sample = X */
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

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double tsample_expect[SIZE_TRANSFORM] = {1, -1, -0.5, RESOLUTION_X};
    double tline_expect[SIZE_TRANSFORM] = {-1, 26, -0.5, RESOLUTION_Y};
    int size_line_expect = number_pixel_y;
    int size_sample_expect = number_pixel_x;

    if (map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	map_info.origin_code != origin_code ||
	map_info.pix_reg_code != pix_reg_code) {
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < SIZE_TRANSFORM; i++) {
      if (MTKm_CMP_NE_DBL(map_info.tline[i],tline_expect[i]) ||
	  MTKm_CMP_NE_DBL(map_info.tsample[i],tsample_expect[i])) {
	fprintf(stderr,"map_info.tline[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tline[i], tline_expect[i]);
	fprintf(stderr,"map_info.tsample[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tsample[i], tsample_expect[i]);
	fprintf(stderr,"Unexpected result(test2).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 4                                                      */
  /* ------------------------------------------------------------------ */

  origin_code = MTKe_ORIGIN_UR;  /* Line = X ; Sample = Y */
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

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double tline_expect[SIZE_TRANSFORM] = {-1, 16, -0.5, RESOLUTION_X};
    double tsample_expect[SIZE_TRANSFORM] = {-1, 26, -0.5, RESOLUTION_Y};

    int size_line_expect = number_pixel_x;
    int size_sample_expect = number_pixel_y;

    if (map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	map_info.origin_code != origin_code ||
	map_info.pix_reg_code != pix_reg_code) {
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < SIZE_TRANSFORM; i++) {
      if (MTKm_CMP_NE_DBL(map_info.tline[i],tline_expect[i]) ||
	  MTKm_CMP_NE_DBL(map_info.tsample[i],tsample_expect[i])) {
	fprintf(stderr,"map_info.tline[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tline[i], tline_expect[i]);
	fprintf(stderr,"map_info.tsample[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tsample[i], tsample_expect[i]);
	fprintf(stderr,"Unexpected result(test2).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 5                                                      */
  /* ------------------------------------------------------------------ */

  pix_reg_code = MTKe_PIX_REG_CORNER;
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

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double tline_expect[SIZE_TRANSFORM] = {-1, 16, 0.0, RESOLUTION_X};
    double tsample_expect[SIZE_TRANSFORM] = {-1, 26, 0.0, RESOLUTION_Y};
    int size_line_expect = number_pixel_x;
    int size_sample_expect = number_pixel_y;

    if (map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	map_info.origin_code != origin_code ||
	map_info.pix_reg_code != pix_reg_code) {
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < SIZE_TRANSFORM; i++) {
      if (MTKm_CMP_NE_DBL(map_info.tline[i],tline_expect[i]) ||
	  MTKm_CMP_NE_DBL(map_info.tsample[i],tsample_expect[i])) {
	fprintf(stderr,"map_info.tline[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tline[i], tline_expect[i]);
	fprintf(stderr,"map_info.tsample[%d] = %20.20g (expected %20.20g)\n",
		i,map_info.tsample[i], tsample_expect[i]);
	fprintf(stderr,"Unexpected result(test5).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Resolution_x <= 0.0                                */
  /*                 Resolution_y <= 0.0                                */
  /* ------------------------------------------------------------------ */

  resolution_x = 0.0;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  resolution_x = 0.1;

  resolution_y = 0.0;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  resolution_y = 0.1;

  /* ------------------------------------------------------------------ */
  /* Argument check: Number_pixel_x < 1                                 */
  /*                 Number_pixel_y < 1                                 */
  /* ------------------------------------------------------------------ */

  number_pixel_x = 0;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  number_pixel_x = 1;

  number_pixel_y = 0;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  number_pixel_y = 1;

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument_check: Unsupported Origin_code                            */
  /* ------------------------------------------------------------------ */

  origin_code = -1;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  origin_code = 0;

  /* ------------------------------------------------------------------ */
  /* Argument_check: Unsupported Pix_reg_code                           */
  /* ------------------------------------------------------------------ */

  pix_reg_code = -1;
  status = MtkGenericMapInfo(min_x,
			     min_y,
			     resolution_x,
			     resolution_y,
			     number_pixel_x,
			     number_pixel_y,
			     origin_code,
			     pix_reg_code,
			     &map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  pix_reg_code = 0;

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
