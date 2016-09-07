/*===========================================================================
=                                                                           =
=                           MtkGCTPProjInfoRead_test                        =
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
  MTKt_GCTPProjInfo proj_info = MTKT_GCTPPROJINFO_INIT;
  char *filename = "MapQuery/test/in/proj_map_info.txt";
  int i;
  
  MTK_PRINT_STATUS(cn,"Testing MtkGCTPProjInfoRead");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfoRead(filename, 
			       &proj_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkGCTPProjInfoRead(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    double proj_param_expect[15] = {0.01,
				    0.02,
				    0.03,
				    0.04,
				    0.0,
				    0.06,
				    0.07,
				    0.08,
				    0.09,
				    0.10,
				    0.11,
				    0.12,
				    0.13,
				    0.14,
				    0.15};
    int proj_code_expect = 7;
    int zone_code_expect = 8;
    int sphere_code_expect = 9;

    if (proj_info.proj_code != proj_code_expect ||
	proj_info.sphere_code != sphere_code_expect ||
	proj_info.zone_code != zone_code_expect) {
      fprintf(stderr,"proj_info.proj_code = %d (expected %d)\n",
	      proj_info.proj_code, proj_code_expect);
      fprintf(stderr,"proj_info.sphere_code = %d (expected %d)\n",
	      proj_info.sphere_code, sphere_code_expect);
      fprintf(stderr,"proj_info.zone_code = %d (expected %d)\n",
	      proj_info.zone_code, zone_code_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < 15; i++) {
      if (MTKm_CMP_NE_DBL(proj_info.proj_param[i], proj_param_expect[i])) {
	fprintf(stderr,"proj_info.proj_param[%d] = %20.20g (expected %20.20g)\n",
		i,proj_info.proj_param[i], proj_param_expect[i]);
	fprintf(stderr,"Unexpected result(test1).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Filename == NULL                                   */
  /*                 Invalid filename.                                  */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfoRead(NULL, 
			       &proj_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  filename = "ProjQuery/test/in/proj_map_info14.txt";
  status = MtkGCTPProjInfoRead(filename, 
			       &proj_info);
  if (status != MTK_FILE_OPEN) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Proj_info == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfoRead(filename, 
			       NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: proj_code not found                                */
  /*                 proj_code == UTM && utm_zone not found             */
  /*                 sphere_code not found                              */
  /* ------------------------------------------------------------------ */

  filename = "MapQuery/test/in/proj_map_info14.txt";
  status = MtkGCTPProjInfoRead(filename, 
			       &proj_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1a)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info15.txt";
  status = MtkGCTPProjInfoRead(filename, 
			       &proj_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1b)\n");
    error = MTK_TRUE;
  }

  filename = "MapQuery/test/in/proj_map_info16.txt";
  status = MtkGCTPProjInfoRead(filename, 
			       &proj_info);
  if (status != MTK_NOT_FOUND) {
    fprintf(stderr,"Unexpected status(1c)\n");
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
