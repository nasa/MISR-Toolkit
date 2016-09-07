/*===========================================================================
=                                                                           =
=                           MtkGCTPProjInfo_test                          =
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
  int proj_code = 1;
  int sphere_code = 2;
  int zone_code = 3;
  double proj_param[15];
  int i;

  MTK_PRINT_STATUS(cn,"Testing MtkGCTPProjInfo");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  for (i = 0 ; i < 15; i++) {
    proj_param[i] = i/10.0;
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
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    if (proj_info.proj_code != proj_code ||
	proj_info.sphere_code != sphere_code ||
	proj_info.zone_code != zone_code) {
      fprintf(stderr,"proj_info.proj_code = %d (expected %d)\n",
	      proj_info.proj_code, proj_code);
      fprintf(stderr,"proj_info.sphere_code = %d (expected %d)\n",
	      proj_info.sphere_code, sphere_code);
      fprintf(stderr,"proj_info.zone_code = %d (expected %d)\n",
	      proj_info.zone_code, zone_code);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (i = 0 ; i < 15; i++) {
      if (proj_info.proj_param[i] != proj_param[i]) {
	fprintf(stderr,"proj_info.proj_param[%d] = %20.20g (expected %20.20g)\n",
		i,proj_info.proj_param[i], proj_param[i]);
	fprintf(stderr,"Unexpected result(test1).\n");
	error = MTK_TRUE;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /* ------------------------------------------------------------------ */

  status = MtkGCTPProjInfo(proj_code,
			   sphere_code,
			   zone_code,
			   proj_param,
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
