/*===========================================================================
=                                                                           =
=                           MtkGenericMapInfoRead_test                      =
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

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_GenericMapInfo map_info = MTKT_GENERICMAPINFO_INIT;
  char *filename = "MapQuery/test/in/proj_map_info.txt";
  
  MTK_PRINT_STATUS(cn,"Testing MtkGenericMapInfoRead");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double min_x_expect = 1.0;
    double min_y_expect = 2.0;
    double resolution_x_expect = 3.0;
    double resolution_y_expect = 4.0;
    int size_line_expect = 5;
    int size_sample_expect = 6;
    double max_x_expect = 16.0;
    double max_y_expect = 26.0;
    MTKt_OriginCode origin_code_expect = MTKe_ORIGIN_LL;
    MTKt_PixRegCode pix_reg_code_expect = MTKe_PIX_REG_CORNER;

    if (MTKm_CMP_NE_DBL(map_info.min_x, min_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.min_y, min_y_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	MTKm_CMP_NE_DBL(map_info.resolution_x, resolution_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.resolution_y, resolution_y_expect) ||
	map_info.origin_code != origin_code_expect ||
	map_info.pix_reg_code != pix_reg_code_expect) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x_expect);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y_expect);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x_expect);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code_expect);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info10.txt";

  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double min_x_expect = 1.0;
    double min_y_expect = 2.0;
    double resolution_x_expect = 3.0;
    double resolution_y_expect = 4.0;
    int size_line_expect = 6;
    int size_sample_expect = 5;
    double max_x_expect = 16.0;
    double max_y_expect = 26.0;
    MTKt_OriginCode origin_code_expect = MTKe_ORIGIN_UL;
    MTKt_PixRegCode pix_reg_code_expect = MTKe_PIX_REG_CORNER;
  
    if (MTKm_CMP_NE_DBL(map_info.min_x, min_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.min_y, min_y_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	MTKm_CMP_NE_DBL(map_info.resolution_x, resolution_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.resolution_y, resolution_y_expect) ||
	map_info.origin_code != origin_code_expect ||
	map_info.pix_reg_code != pix_reg_code_expect) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x_expect);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y_expect);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x_expect);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code_expect);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code_expect);
      fprintf(stderr,"Unexpected result(test2).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 3                                                      */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info11.txt";

  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double min_x_expect = 1.0;
    double min_y_expect = 2.0;
    double resolution_x_expect = 3.0;
    double resolution_y_expect = 4.0;
    int size_line_expect = 5;
    int size_sample_expect = 6;
    double max_x_expect = 16.0;
    double max_y_expect = 26.0;
    MTKt_OriginCode origin_code_expect = MTKe_ORIGIN_UR;
    MTKt_PixRegCode pix_reg_code_expect = MTKe_PIX_REG_CORNER;
  
    if (MTKm_CMP_NE_DBL(map_info.min_x, min_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.min_y, min_y_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	MTKm_CMP_NE_DBL(map_info.resolution_x, resolution_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.resolution_y, resolution_y_expect) ||
	map_info.origin_code != origin_code_expect ||
	map_info.pix_reg_code != pix_reg_code_expect) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x_expect);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y_expect);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x_expect);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code_expect);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code_expect);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 4                                                      */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info12.txt";

  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double min_x_expect = 1.0;
    double min_y_expect = 2.0;
    double resolution_x_expect = 3.0;
    double resolution_y_expect = 4.0;
    int size_line_expect = 5;
    int size_sample_expect = 6;
    double max_x_expect = 16.0;
    double max_y_expect = 26.0;
    MTKt_OriginCode origin_code_expect = MTKe_ORIGIN_LL;
    MTKt_PixRegCode pix_reg_code_expect = MTKe_PIX_REG_CENTER;
  
    if (MTKm_CMP_NE_DBL(map_info.min_x, min_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.min_y, min_y_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	MTKm_CMP_NE_DBL(map_info.resolution_x, resolution_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.resolution_y, resolution_y_expect) ||
	map_info.origin_code != origin_code_expect ||
	map_info.pix_reg_code != pix_reg_code_expect) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x_expect);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y_expect);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x_expect);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code_expect);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code_expect);
      fprintf(stderr,"Unexpected result(test4).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 5                                                      */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info13.txt";

  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGenericMapInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double min_x_expect = 1.0;
    double min_y_expect = 2.0;
    double resolution_x_expect = 3.0;
    double resolution_y_expect = 4.0;
    int size_line_expect = 6;
    int size_sample_expect = 5;
    double max_x_expect = 16.0;
    double max_y_expect = 26.0;
    MTKt_OriginCode origin_code_expect = MTKe_ORIGIN_LR;
    MTKt_PixRegCode pix_reg_code_expect = MTKe_PIX_REG_CORNER;
  
    if (MTKm_CMP_NE_DBL(map_info.min_x, min_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.min_y, min_y_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_x, max_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.max_y, max_y_expect) ||
	map_info.size_line != size_line_expect ||
	map_info.size_sample != size_sample_expect ||
	MTKm_CMP_NE_DBL(map_info.resolution_x, resolution_x_expect) ||
	MTKm_CMP_NE_DBL(map_info.resolution_y, resolution_y_expect) ||
	map_info.origin_code != origin_code_expect ||
	map_info.pix_reg_code != pix_reg_code_expect) {
      fprintf(stderr,"map_info.min_x = %20.20g (expected %20.20g)\n",
	      map_info.min_x, min_x_expect);
      fprintf(stderr,"map_info.min_y = %20.20g (expected %20.20g)\n",
	      map_info.min_y, min_y_expect);
      fprintf(stderr,"map_info.max_x = %20.20g (expected %20.20g)\n",
	      map_info.max_x, max_x_expect);
      fprintf(stderr,"map_info.max_y = %20.20g (expected %20.20g)\n",
	      map_info.max_y, max_y_expect);
      fprintf(stderr,"map_info.size_line = %d (expected %d)\n",
	      map_info.size_line, size_line_expect);
      fprintf(stderr,"map_info.size_sample = %d (expected %d)\n",
	      map_info.size_sample, size_sample_expect);
      fprintf(stderr,"map_info.resolution_x = %20.20g (expected %20.20g)\n",
	      map_info.resolution_x, resolution_x_expect);
      fprintf(stderr,"map_info.resolution_y = %20.20g (expected %20.20g)\n",
	      map_info.resolution_y, resolution_y_expect);
      fprintf(stderr,"map_info.origin_code = %d (expected %d)\n",
	      map_info.origin_code, origin_code_expect);
      fprintf(stderr,"map_info.pix_reg_code = %d (expected %d)\n",
	      map_info.pix_reg_code, pix_reg_code_expect);
      fprintf(stderr,"Unexpected result(test5).\n");
      error = MTK_TRUE;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Filename == NULL                                   */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfoRead(NULL, 
				 &map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkGenericMapInfoRead(filename, 
				 NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: min_corner_x not found                             */
  /*                 min_corner_y not found                             */
  /*                 resolution_x not found                             */
  /*                 resolution_y not found                             */
  /*                 number_pixel_x not found                           */
  /*                 number_pixel_y not found                           */
  /*                 origin_code not found                              */
  /*                 pix_reg_code not found                             */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info2.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info3.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info4.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info5.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info6.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info7.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info8.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info9.txt";
  status = MtkGenericMapInfoRead(filename, 
				 &map_info);
  if (status != MTK_NOT_FOUND) {
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
