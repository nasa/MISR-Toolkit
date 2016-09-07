/*===========================================================================
=                                                                           =
=                           MtkApplyRegression_test                         =
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

#define SOURCE_NUMBER_LINE 4
#define SOURCE_NUMBER_SAMPLE 8

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer source = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer source_mask = MTKT_DATABUFFER_INIT;
  MTKt_RegressionCoeff regression_coeff = MTKT_REGRESSION_COEFF_INIT;
  MTKt_MapInfo source_map_info = MTKT_MAPINFO_INIT;
  MTKt_MapInfo regression_coeff_map_info = MTKT_MAPINFO_INIT;
  MTKt_Region region = MTKT_REGION_INIT;
				/* Region structure */
  MTKt_DataBuffer regressed = MTKT_DATABUFFER_INIT;
				/* Output data. */
  MTKt_DataBuffer regressed_mask = MTKT_DATABUFFER_INIT;
				/* Output data mask */
  int iline, isample;
  
  MTK_PRINT_STATUS(cn,"Testing MtkApplyRegression");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(SOURCE_NUMBER_LINE, SOURCE_NUMBER_SAMPLE, 
				 MTKe_float, &source);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(SOURCE_NUMBER_LINE, SOURCE_NUMBER_SAMPLE, 
				 MTKe_uint8, &source_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  for (iline = 0 ; iline < SOURCE_NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < SOURCE_NUMBER_SAMPLE ; isample++) {
      source.data.f[iline][isample] = -iline * 0.1 - isample * 0.001;
      source_mask.data.u8[iline][isample] = 1;
    }
  }
  source_mask.data.u8[2][3] = 0;

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

  status = MtkSnapToGrid(39, 4400, region, &source_map_info);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkSnapToGrid(2)\n");
    error = MTK_TRUE;
  }
  if (source_map_info.nline != SOURCE_NUMBER_LINE ||
      source_map_info.nsample != SOURCE_NUMBER_SAMPLE) {
    fprintf(stderr,"source_map_info.nline = %d (expected %d)\n",
	    source_map_info.nline, SOURCE_NUMBER_LINE);
    fprintf(stderr,"source_map_info.nsample = %d (expected %d)\n",
	    source_map_info.nsample, SOURCE_NUMBER_SAMPLE);
    fprintf(stderr,"Unexpected result(test1).\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkApplyRegression(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    uint8 regressed_mask_expect[SOURCE_NUMBER_LINE][SOURCE_NUMBER_SAMPLE] = {
      {1, 1, 1, 1, 1, 1, 1, 1 },
      {1, 1, 1, 1, 1, 1, 1, 1 },
      {1, 1, 1, 0, 1, 1, 1, 1 },
      {1, 1, 1, 1, 1, 1, 1, 1 }
    };

    float regressed_expect[SOURCE_NUMBER_LINE][SOURCE_NUMBER_SAMPLE] = {
{0.36712500452995300293, 0.36664026975631713867, 0.36615452170372009277, 0.36566770076751708984, 0.36518001556396484375, 0.36469125747680664062, 0.36420151591300964355, 0.36371076107025146484 },
{0.31370002031326293945, 0.31311529874801635742, 0.31252953410148620605, 0.31194275617599487305, 0.3113549649715423584, 0.3107662200927734375, 0.31017646193504333496, 0.30958575010299682617 },
{0.25027501583099365234, 0.24959027767181396484, 0.2489045560359954834,                    0, 0.24752999842166900635, 0.24684123694896697998, 0.24615146219730377197, 0.24546076357364654541 },
{0.17684996128082275391, 0.17606526613235473633, 0.17527952790260314941, 0.17449274659156799316, 0.17370501160621643066, 0.17291623353958129883, 0.17212653160095214844, 0.17133575677871704102 }
    };

    if (regressed.nline != SOURCE_NUMBER_LINE ||
	regressed.nsample != SOURCE_NUMBER_SAMPLE ||
	regressed_mask.nline != SOURCE_NUMBER_LINE ||
	regressed_mask.nsample != SOURCE_NUMBER_SAMPLE) {
      fprintf(stderr,"regressed.nline = %d (expected %d)\n",
	      regressed.nline, SOURCE_NUMBER_LINE);
      fprintf(stderr,"regressed.nsample = %d (expected %d)\n",
	      regressed.nsample, SOURCE_NUMBER_SAMPLE);
      fprintf(stderr,"regressed_mask.nline = %d (expected %d)\n",
	      regressed_mask.nline, SOURCE_NUMBER_LINE);
      fprintf(stderr,"regressed_mask.nsample = %d (expected %d)\n",
	      regressed_mask.nsample, SOURCE_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (iline = 0 ; iline < SOURCE_NUMBER_LINE ; iline++) {
      for (isample = 0 ; isample < SOURCE_NUMBER_SAMPLE; isample++) {
	if (MTKm_CMP_NE_FLT(regressed.data.f[iline][isample],
			    regressed_expect[iline][isample]) ||
	    regressed_mask.data.u8[iline][isample] != 
	    regressed_mask_expect[iline][isample]) {
	  fprintf(stderr,"regressed.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, regressed.data.f[iline][isample],
		  regressed_expect[iline][isample]);
	  fprintf(stderr,"regressed_mask.data.u8[%d][%d] = %d (expected %d)\n",
		  iline, isample, regressed_mask.data.u8[iline][isample],
		  regressed_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    if (error) {
      fprintf(stderr,"regressed_mask:\n");
      for (iline = 0 ; iline < SOURCE_NUMBER_LINE ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < SOURCE_NUMBER_SAMPLE; isample++) {
	  fprintf(stderr,"%d, ",regressed_mask.data.u8[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"regressed:\n");
      for (iline = 0 ; iline < SOURCE_NUMBER_LINE ; iline++) {
	fprintf(stderr,"{");
	for (isample = 0 ; isample < SOURCE_NUMBER_SAMPLE; isample++) {
	  fprintf(stderr,"%20.20g, ",regressed.data.f[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Source == NULL                                     */
  /*                 Source->nline < 1                                  */
  /*                 Source->nsample < 1                                */
  /*                 Source->datatype = MTKe_float                      */
  /* ------------------------------------------------------------------ */

  status = MtkApplyRegression(NULL,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source.nline = 0;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nline = SOURCE_NUMBER_LINE;

  source.nsample = 0;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nsample = SOURCE_NUMBER_SAMPLE;

  source.datatype++;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Source_mask == NULL                                */
  /*                 Source_mask->nline != Source->nline                */
  /*                 Source_mask->nsample != Source->nsample            */
  /*                 Source_mask->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  status = MtkApplyRegression(&source,
			      NULL, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source_mask.nline++;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nline--;

  source_mask.nsample++;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nsample--;

  source_mask.datatype++;
  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
			      &regressed_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Regressed == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      NULL,
			      &regressed_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regressed_mask == NULL                             */
  /* ------------------------------------------------------------------ */

  status = MtkApplyRegression(&source,
			      &source_mask, 
			      &source_map_info, 
			      &regression_coeff,
			      &regression_coeff_map_info,
			      &regressed,
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
