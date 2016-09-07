/*===========================================================================
=                                                                           =
=                           MtkResampleRegressionCoeff_test                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrRegression.h"
#include "MisrSetRegion.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_DBL(x,y) (fabs((x)-(y)) > DBL_EPSILON * 10 * fabs(x))
#define MTKm_CMP_NE_FLT(x,y) (fabs((x)-(y)) > FLT_EPSILON * 10 * fabs(x))

#define NUMBER_LINE_EXPECT 4
#define NUMBER_SAMPLE_EXPECT 8

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_RegressionCoeff regression_coeff = MTKT_REGRESSION_COEFF_INIT;
  MTKt_RegressionCoeff regression_coeff_out = MTKT_REGRESSION_COEFF_INIT;
  MTKt_MapInfo target_map_info = MTKT_MAPINFO_INIT;
  MTKt_MapInfo regression_coeff_map_info = MTKT_MAPINFO_INIT;
  MTKt_Region region = MTKT_REGION_INIT;
				/* Region structure */
  int iline, isample;
  
  MTK_PRINT_STATUS(cn,"Testing MtkResampleRegressionCoeff");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByPathBlockRange(39, 51, 52, &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByPathBlockRange(1)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(39, 17600, region, &regression_coeff_map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(1)\n");
    error = MTK_TRUE;
  }

  status = MtkRegressionCoeffAllocate(regression_coeff_map_info.nline,
				      regression_coeff_map_info.nsample,
				      &regression_coeff);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkRegressionCoeffAllocate(1)\n");
    error = MTK_TRUE;
  }

  for (iline = 0 ; iline < regression_coeff_map_info.nline ; iline++) {
    for (isample = 0 ; isample < regression_coeff_map_info.nsample ; isample++) {
      regression_coeff.intercept.data.f[iline][isample] = iline * 0.1 + isample * 0.001;
      regression_coeff.slope.data.f[iline][isample] = 2.0 * 
	regression_coeff.intercept.data.f[iline][isample];
      regression_coeff.correlation.data.f[iline][isample] = 3.0 * 
	regression_coeff.intercept.data.f[iline][isample];

      regression_coeff.valid_mask.data.u8[iline][isample] = 1.0;
    }
  }

  status = MtkSetRegionByLatLonExtent(49.435794, -112.637708,
				      17600-10,
				      17600*2-10,
				      "meters",
				      &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByLatLonExtent(1)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(39, 4400, region, &target_map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(2)\n");
    error = MTK_TRUE;
  }
  if (target_map_info.nline != NUMBER_LINE_EXPECT ||
      target_map_info.nsample != NUMBER_SAMPLE_EXPECT) {
    fprintf(stderr,"target_map_info.nline = %d (expected %d)\n",
	    target_map_info.nline, NUMBER_LINE_EXPECT);
    fprintf(stderr,"target_map_info.nsample = %d (expected %d)\n",
	    target_map_info.nsample, NUMBER_SAMPLE_EXPECT);
    fprintf(stderr,"Unexpected result(test1).\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkResampleRegressionCoeff(&regression_coeff,
				      &regression_coeff_map_info, 
				      &target_map_info, 
				      &regression_coeff_out);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkResampleRegressionCoeff(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    uint8 valid_mask_expect[NUMBER_LINE_EXPECT][NUMBER_SAMPLE_EXPECT] = {
      {1,1,1,1,1,1,1,1},
      {1,1,1,1,1,1,1,1},
      {1,1,1,1,1,1,1,1},
      {1,1,1,1,1,1,1,1}
    };

    float intercept_expect[NUMBER_LINE_EXPECT][NUMBER_SAMPLE_EXPECT] = {
{0.36712500452995300293, 0.36737501621246337891, 0.36762502789497375488, 0.36787495017051696777, 0.36812502145767211914, 0.36837500333786010742, 0.3686250150203704834, 0.36887502670288085938 },
{0.39212501049041748047, 0.39237505197525024414, 0.39262503385543823242, 0.3928750157356262207, 0.39312496781349182129, 0.39337497949600219727, 0.39362496137619018555, 0.39387500286102294922 },
{0.4171250462532043457, 0.41737505793571472168, 0.41762509942054748535, 0.41787502169609069824, 0.41812500357627868652, 0.4183749854564666748, 0.41862493753433227539, 0.41887503862380981445 },
{0.44212496280670166016, 0.44237500429153442383, 0.4426250159740447998, 0.44287499785423278809, 0.44312500953674316406, 0.44337499141693115234, 0.44362503290176391602, 0.4438750147819519043 }
    };

    float slope_expect[NUMBER_LINE_EXPECT][NUMBER_SAMPLE_EXPECT];
    float correlation_expect[NUMBER_LINE_EXPECT][NUMBER_SAMPLE_EXPECT];

    for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
      for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	slope_expect[iline][isample] = 2.0 * intercept_expect[iline][isample];
	correlation_expect[iline][isample] = 3.0 * intercept_expect[iline][isample];
      }
    }

    if (regression_coeff_out.valid_mask.nline != target_map_info.nline ||
	regression_coeff_out.valid_mask.nsample != target_map_info.nsample ||
	regression_coeff_out.intercept.nline != target_map_info.nline ||
	regression_coeff_out.intercept.nsample != target_map_info.nsample ||
	regression_coeff_out.slope.nline != target_map_info.nline ||
	regression_coeff_out.slope.nsample != target_map_info.nsample ||
	regression_coeff_out.correlation.nline != target_map_info.nline ||
	regression_coeff_out.correlation.nsample != target_map_info.nsample) {
      fprintf(stderr,"regression_coeff_out.valid_mask.nline = %d (expected %d)\n",
	      regression_coeff_out.valid_mask.nline,target_map_info.nline);
      fprintf(stderr,"regression_coeff_out.valid_mask.nsample = %d (expected %d)\n",
	      regression_coeff_out.valid_mask.nsample,target_map_info.nsample);
      fprintf(stderr,"regression_coeff_out.intercept.nline = %d (expected %d)\n",
	      regression_coeff_out.intercept.nline,target_map_info.nline);
      fprintf(stderr,"regression_coeff_out.intercept.nsample = %d (expected %d)\n",
	      regression_coeff_out.intercept.nsample,target_map_info.nsample);
      fprintf(stderr,"regression_coeff_out.slope.nline = %d (expected %d)\n",
	      regression_coeff_out.slope.nline,target_map_info.nline);
      fprintf(stderr,"regression_coeff_out.slope.nsample = %d (expected %d)\n",
	      regression_coeff_out.slope.nsample,target_map_info.nsample);
      fprintf(stderr,"regression_coeff_out.correlation.nline = %d (expected %d)\n",
	      regression_coeff_out.correlation.nline,target_map_info.nline);
      fprintf(stderr,"regression_coeff_out.correlation.nsample = %d (expected %d)\n",
	      regression_coeff_out.correlation.nsample,target_map_info.nsample);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
      for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	if (regression_coeff_out.valid_mask.data.u8[iline][isample] != 
	    valid_mask_expect[iline][isample] ||
	    MTKm_CMP_NE_FLT(regression_coeff_out.intercept.data.f[iline][isample],
			    intercept_expect[iline][isample]) ||
	    MTKm_CMP_NE_FLT(regression_coeff_out.slope.data.f[iline][isample],
			    slope_expect[iline][isample]) ||
	    MTKm_CMP_NE_FLT(regression_coeff_out.correlation.data.f[iline][isample],
			    correlation_expect[iline][isample])) {
	  fprintf(stderr,"valid_mask[%d][%d] = %d (expected %d)\n",
		  iline, isample,
		  regression_coeff_out.valid_mask.data.u8[iline][isample],
		  valid_mask_expect[iline][isample]);
	  fprintf(stderr,"intercept[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample,
		  regression_coeff_out.intercept.data.f[iline][isample],
		  intercept_expect[iline][isample]);
	  fprintf(stderr,"slope[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample,
		  regression_coeff_out.slope.data.f[iline][isample],
		  slope_expect[iline][isample]);
	  fprintf(stderr,"correlation[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample,
		  regression_coeff_out.correlation.data.f[iline][isample],
		  correlation_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    if (error) {
      fprintf(stderr,"valid_mask:\n");
      for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	  fprintf(stderr,"%d, ",regression_coeff_out.valid_mask.data.u8[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"intercept:\n");
      for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff_out.intercept.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"slope:\n");
      for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff_out.slope.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"correlation:\n");
      for (iline = 0 ; iline < NUMBER_LINE_EXPECT ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < NUMBER_SAMPLE_EXPECT; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff_out.correlation.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff == NULL                           */
  /* ------------------------------------------------------------------ */

  status = MtkResampleRegressionCoeff(NULL,
				      &regression_coeff_map_info, 
				      &target_map_info, 
				      &regression_coeff_out);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff_out == NULL                       */
  /* ------------------------------------------------------------------ */

  status = MtkResampleRegressionCoeff(&regression_coeff,
				      &regression_coeff_map_info, 
				      &target_map_info, 
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
