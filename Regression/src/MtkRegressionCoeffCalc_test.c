/*===========================================================================
=                                                                           =
=                           MtkRegressionCoeffCalc_test                     =
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

#define MTKm_CMP_NE_FLT(x,y) (fabs((x)-(y)) > FLT_EPSILON * 100 * fabs(x))

#define NUMBER_LINE 40
#define NUMBER_SAMPLE 60

#define REGRESS_NUMBER_LINE 4
#define REGRESS_NUMBER_SAMPLE 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer data1 = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer data2 = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer data2_sigma = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer valid_mask1 = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer valid_mask2 = MTKT_DATABUFFER_INIT;
  MTKt_RegressionCoeff regression_coeff = MTKT_REGRESSION_COEFF_INIT;
  MTKt_MapInfo map_info = MTKT_MAPINFO_INIT;
  MTKt_MapInfo regression_coeff_map_info = MTKT_MAPINFO_INIT;
  MTKt_Region region = MTKT_REGION_INIT;
  int regression_size_factor = 10;
  int iline, isample;
  double center_lat = 1.0;
  double center_lon = 2.0;
  int path = 192;
  
  MTK_PRINT_STATUS(cn,"Testing MtkRegressionCoeffCalc");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_float,
				 &data1);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_float,
				 &data2);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(2)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_float,
				 &data2_sigma);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(3)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_uint8,
				 &valid_mask1);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(4)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_uint8,
				 &valid_mask2);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(5)\n");
    error = MTK_TRUE;
  }

  for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
      data1.data.f[iline][isample] = 
	cos(iline/(double)NUMBER_LINE) + sin(isample/(double)NUMBER_SAMPLE);
      data2.data.f[iline][isample] = 
	cos((iline+1)/(double)NUMBER_LINE) + sin((isample+1)/(double)NUMBER_SAMPLE);
      data2_sigma.data.f[iline][isample] = 0.1;
      valid_mask1.data.u8[iline][isample] = 1;
      valid_mask2.data.u8[iline][isample] = 1;
    }
  }
  data2_sigma.data.f[11][44] = 0.2;
  valid_mask1.data.u8[22][44] = 0;
  for (iline = 30 ; iline < 40 ; iline++) {
    for (isample = 20 ; isample < 40 ; isample++) {
      valid_mask1.data.u8[iline][isample] = 0;
    }
  }

  for (iline = 20 ; iline < 30 ; iline++) {
    for (isample = 0 ; isample < 20 ; isample++) {
      valid_mask2.data.u8[iline][isample] = 0;
    }
  }

  /* Trigger divide-by-zero condition. */
  for (iline = 10 ; iline < 20 ; iline++) {
    for (isample = 50 ; isample < 60 ; isample++) {
      data1.data.f[iline][isample] = 1.0;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Setup map information.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkSetRegionByLatLonExtent(center_lat,center_lon,NUMBER_LINE,
				      NUMBER_SAMPLE,
				      "1100m", &region);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSetRegionByPathBlockRange(1)\n");
    error = MTK_TRUE;
  }

  status = MtkSnapToGrid(path, 1100, region, &map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(1)\n");
    error = MTK_TRUE;
  }
  map_info.nline = NUMBER_LINE;
  map_info.nsample = NUMBER_SAMPLE;

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkRegressionCoeffCalc(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    uint8 regression_coeff_valid_mask_expect[REGRESS_NUMBER_LINE][REGRESS_NUMBER_SAMPLE] = {
      { 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 0 },
      { 0, 0, 1, 1, 1, 1 },
      { 1, 1, 0, 0, 1, 1 }
    };

    float regression_coeff_slope_expect[REGRESS_NUMBER_LINE][REGRESS_NUMBER_SAMPLE] = {
      { 1.004746689150492811, 1.0023076235687278235, 1.0000685741070258761, 0.99802329622236596318, 0.99630024024047858511, 0.99539591240097091696 },
      { 1.0133927538706966054,  1.01182212755445744, 1.0109349321664795607, 1.0109316109316477394, 1.0120569831322761001, 0.0 },
      {                    0,                    0, 1.0125861488435443647, 1.0127412406963378633, 1.0137069975374357611, 1.0160554455488057801 },
      { 1.0113165970772881597, 1.0104079897908466723,                    0,                    0, 1.0102886296433721824, 1.0114481585857448831 },
    };

    float regression_coeff_intercept_expect[REGRESS_NUMBER_LINE][REGRESS_NUMBER_SAMPLE] = {
      { 0.0084210625585166860169, 0.010182339135712887429, 0.012019163040735428294, 0.01381858557411450375, 0.015235238818814457115, 0.015209924059709410693 },
      { -0.0060358805216385383319, -0.0068545434678650733404, -0.0084449403608851739134, -0.011371609461255484733, -0.016346094391744687285, 0.0 },
      {                    0,                    0, -0.014629896093502091359, -0.018014331757920366839, -0.022853229762833562888, -0.030207785428246956144 },
      { -0.01073896401253537046, -0.012250178833004240694,                    0,                    0, -0.020576073214393898747, -0.025453853260072344111 }
    };

    float regression_coeff_correlation_expect[REGRESS_NUMBER_LINE][REGRESS_NUMBER_SAMPLE] = {
      { 0.99934297414409023474, 0.99928795471177389587, 0.99918340822729889705, 0.99900185094244475792, 0.9986843323263598915, 0.99809560465721391953 },
      { 0.99962393444752439819, 0.99957791288395669849, 0.99951124801965518518, 0.99941574678638500639, 0.99928633308945746805, 0.0 },
      {                    0,                    0, 0.99977935201797385467, 0.99973992929280053321, 0.99969557132516206899, 0.99965098835694332635 },
      { 0.99993891661586842279, 0.99992298908503629562,                    0,                    0, 0.99986376883572836149, 0.99984450999226093248 }
    };

    int pp_nline_expect = 12;
    int pp_nsample_expect = 51;
    int resolution_expect = 11000;
    int pp_nblock_expect = 180;

    if (regression_coeff_map_info.nline != REGRESS_NUMBER_LINE ||
	regression_coeff_map_info.nsample != REGRESS_NUMBER_SAMPLE ||
	regression_coeff_map_info.pp.nline != pp_nline_expect ||
	regression_coeff_map_info.pp.nsample != pp_nsample_expect ||
	regression_coeff_map_info.resolution != resolution_expect ||
	regression_coeff_map_info.pp.resolution != resolution_expect ||
	regression_coeff_map_info.pp.nblock != pp_nblock_expect) {
      fprintf(stderr,"regression_coeff_map_info.nline = %d (expected %d)\n",
	     regression_coeff_map_info.nline, REGRESS_NUMBER_LINE);
      fprintf(stderr,"regression_coeff_map_info.nsample = %d (expected %d)\n",
	     regression_coeff_map_info.nsample, REGRESS_NUMBER_SAMPLE);
      fprintf(stderr,"regression_coeff_map_info.pp.nline = %d (expected %d)\n",
	     regression_coeff_map_info.pp.nline, pp_nline_expect);
      fprintf(stderr,"regression_coeff_map_info.pp.nsample = %d (expected %d)\n",
	     regression_coeff_map_info.pp.nsample, pp_nsample_expect);
      fprintf(stderr,"regression_coeff_map_info.resolution = %d (expected %d)\n",
	      regression_coeff_map_info.resolution, resolution_expect);
      fprintf(stderr,"regression_coeff_map_info.pp.resolution = %d (expected %d)\n",
	     regression_coeff_map_info.pp.resolution, resolution_expect);
      fprintf(stderr,"regression_coeff_map_info.pp.nblock = %d (expected %d)\n",
	     regression_coeff_map_info.pp.nblock, pp_nblock_expect);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    if (regression_coeff.valid_mask.nline != REGRESS_NUMBER_LINE ||
	regression_coeff.valid_mask.nsample != REGRESS_NUMBER_SAMPLE ||
	regression_coeff.slope.nline != REGRESS_NUMBER_LINE ||
	regression_coeff.slope.nsample != REGRESS_NUMBER_SAMPLE ||
	regression_coeff.intercept.nline != REGRESS_NUMBER_LINE ||
	regression_coeff.intercept.nsample != REGRESS_NUMBER_SAMPLE ||
	regression_coeff.correlation.nline != REGRESS_NUMBER_LINE ||
	regression_coeff.correlation.nsample != REGRESS_NUMBER_SAMPLE) {
      fprintf(stderr,"regression_coeff.valid_mask.nline = %d (expected %d)\n",
	     regression_coeff.valid_mask.nline, REGRESS_NUMBER_LINE);
      fprintf(stderr,"regression_coeff.valid_mask.nsample = %d (expected %d)\n",
	     regression_coeff.valid_mask.nsample, REGRESS_NUMBER_SAMPLE);
      fprintf(stderr,"regression_coeff.slope.nline = %d (expected %d)\n",
	     regression_coeff.slope.nline, REGRESS_NUMBER_LINE);
      fprintf(stderr,"regression_coeff.slope.nsample = %d (expected %d)\n",
	     regression_coeff.slope.nsample, REGRESS_NUMBER_SAMPLE);
      fprintf(stderr,"regression_coeff.intercept.nline = %d (expected %d)\n",
	     regression_coeff.intercept.nline, REGRESS_NUMBER_LINE);
      fprintf(stderr,"regression_coeff.intercept.nsample = %d (expected %d)\n",
	     regression_coeff.intercept.nsample, REGRESS_NUMBER_SAMPLE);
      fprintf(stderr,"regression_coeff.correlation.nline = %d (expected %d)\n",
	     regression_coeff.correlation.nline, REGRESS_NUMBER_LINE);
      fprintf(stderr,"regression_coeff.correlation.nsample = %d (expected %d)\n",
	     regression_coeff.correlation.nsample, REGRESS_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
    
    for (iline = 0 ; iline < regression_coeff.valid_mask.nline ; iline++) {
      for (isample = 0 ; isample < regression_coeff.valid_mask.nsample ; isample++) {
	if (regression_coeff.valid_mask.data.u8[iline][isample] !=
	    regression_coeff_valid_mask_expect[iline][isample]) {
	  fprintf(stderr,"regression_coeff.valid_mask.data.u8[%d][%d] = %d (expected %d)\n",
		 iline, isample, 
		 regression_coeff.valid_mask.data.u8[iline][isample],
		 regression_coeff_valid_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }
    
    for (iline = 0 ; iline < regression_coeff.slope.nline ; iline++) {
      for (isample = 0 ; isample < regression_coeff.slope.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(regression_coeff.slope.data.f[iline][isample],
			   regression_coeff_slope_expect[iline][isample])) {
	  fprintf(stderr,"regression_coeff.slope.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		 iline, isample, 
		 regression_coeff.slope.data.f[iline][isample],
		 regression_coeff_slope_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    for (iline = 0 ; iline < regression_coeff.intercept.nline ; iline++) {
      for (isample = 0 ; isample < regression_coeff.intercept.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(regression_coeff.intercept.data.f[iline][isample],
			   regression_coeff_intercept_expect[iline][isample])) {
	  fprintf(stderr,"regression_coeff.intercept.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		 iline, isample, 
		 regression_coeff.intercept.data.f[iline][isample],
		 regression_coeff_intercept_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    for (iline = 0 ; iline < regression_coeff.correlation.nline ; iline++) {
      for (isample = 0 ; isample < regression_coeff.correlation.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(regression_coeff.correlation.data.f[iline][isample],
			   regression_coeff_correlation_expect[iline][isample])) {
	  fprintf(stderr,"regression_coeff.correlation.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		 iline, isample, 
		 regression_coeff.correlation.data.f[iline][isample],
		 regression_coeff_correlation_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	
	}
      }
    }
    if (error) {
      fprintf(stderr,"valid_mask:\n");
      for (iline = 0 ; iline < regression_coeff.valid_mask.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < regression_coeff.valid_mask.nsample ; isample++) {
	  fprintf(stderr,"%d, ",regression_coeff.valid_mask.data.u8[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"slope:\n");
      for (iline = 0 ; iline < regression_coeff.slope.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < regression_coeff.slope.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff.slope.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"intercept:\n");
      for (iline = 0 ; iline < regression_coeff.intercept.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < regression_coeff.intercept.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff.intercept.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"correlation:\n");
      for (iline = 0 ; iline < regression_coeff.correlation.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < regression_coeff.correlation.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",regression_coeff.correlation.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Data1 == NULL                                      */
  /*                 Data1->nline < 1                                   */
  /*                 Data1->nsample < 1                                 */
  /*                 Data1->datatype = MTKe_float                      */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(NULL,
				  &valid_mask1,
				  &data2,
				  &data2_sigma,
				  &valid_mask2,
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  data1.nline = 0;
  status = MtkRegressionCoeffCalc(&data1,
				  &valid_mask1,
				  &data2,
				  &data2_sigma,
				  &valid_mask2,
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data1.nline = NUMBER_LINE;

  data1.nsample = 0;
  status = MtkRegressionCoeffCalc(&data1,
				  &valid_mask1,
				  &data2,
				  &data2_sigma,
				  &valid_mask2,
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data1.nsample = NUMBER_SAMPLE;

  data1.datatype++;
  status = MtkRegressionCoeffCalc(&data1,
				  &valid_mask1,
				  &data2,
				  &data2_sigma,
				  &valid_mask2,
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data1.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Valid_mask1 == NULL                                */
  /*                 Valid_mask1->nline != Data1->nline                 */
  /*                 Valid_mask1->nsample != Data1->nsample             */
  /*                 Valid_mask1->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  NULL, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  valid_mask1.nline++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask1.nline--;

  valid_mask1.nsample++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask1.nsample--;

  valid_mask1.datatype++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask1.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Data2 == NULL                                      */
  /*                 Data2->nline != Data1->nline                       */
  /*                 Data2->nsample != Data1->nsample                   */
  /*                 Data2->datatype = MTKe_float                      */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  NULL, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  data2.nline++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2.nline--;

  data2.nsample++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2.nsample--;

  data2.datatype++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Data2_sigma == NULL                                */
  /*                 Data2_sigma->nline != Data1->nline                 */
  /*                 Data2_sigma->nsample != Data1->nsample             */
  /*                 Data2_sigma->datatype = MTKe_float                */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  NULL, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  data2_sigma.nline++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2_sigma.nline--;

  data2_sigma.nsample++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2_sigma.nsample--;

  data2_sigma.datatype++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  data2_sigma.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Valid_mask2 == NULL                                */
  /*                 Valid_mask2->nline != Data1->nline                 */
  /*                 Valid_mask2->nsample != Data1->nsample             */
  /*                 Valid_mask2->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  NULL, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  valid_mask2.nline++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask2.nline--;

  valid_mask2.nsample++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask2.nsample--;

  valid_mask2.datatype++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  valid_mask2.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /*                 Map_info->nline != Data1->nline                    */
  /*                 Map_info->nsample != Data1->nsample                */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  NULL,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  map_info.nline++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info.nline--;

  map_info.nsample++;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  map_info.nsample--;

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   Data1->nline % Regression_size_factor != 0                       */
  /*   Data1->nsample in % Regression_size_factor != 0                  */
  /* ------------------------------------------------------------------ */

  regression_size_factor = 60;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  regression_size_factor = 10;

  regression_size_factor = 40;
  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
				  &regression_coeff_map_info);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  regression_size_factor = 10;

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff == NULL                           */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  NULL,
				  &regression_coeff_map_info);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff_map_info == NULL                  */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffCalc(&data1, 
				  &valid_mask1, 
				  &data2, 
				  &data2_sigma, 
				  &valid_mask2, 
				  &map_info,
				  regression_size_factor, 
				  &regression_coeff,
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
