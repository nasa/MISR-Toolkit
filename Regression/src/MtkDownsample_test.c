/*===========================================================================
=                                                                           =
=                           MtkDownsample_test                              =
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
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_FLT(x,y) (fabs((x)-(y)) > FLT_EPSILON * 100 * fabs(x))

#define NUMBER_LINE 40
#define NUMBER_SAMPLE 60

#define RESULT_NUMBER_LINE 4
#define RESULT_NUMBER_SAMPLE 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer source = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer source_mask = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer result = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer result_mask = MTKT_DATABUFFER_INIT;
  int size_factor = 10;
  int iline, isample;
  
  MTK_PRINT_STATUS(cn,"Testing MtkDownsample");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_float,
				 &source);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_uint8,
				 &source_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
      source.data.f[iline][isample] = iline * NUMBER_SAMPLE + isample;
      source_mask.data.u8[iline][isample] = 1;
    }
  }
  source_mask.data.u8[10][30] = 0;
  
  for (iline = 20 ; iline < 30 ; iline++) {
    for (isample = 40 ; isample < 50 ; isample++) {
      source_mask.data.u8[iline][isample] = 0;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDownsample(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    uint8 result_mask_expect[RESULT_NUMBER_LINE][RESULT_NUMBER_SAMPLE] = {
      { 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 0, 1 },
      { 1, 1, 1, 1, 1, 1 }
    };

    float result_expect[RESULT_NUMBER_LINE][RESULT_NUMBER_SAMPLE] = {
      {274.500,      284.500,      294.500,      304.500,      314.500,      324.500},
      {874.500,      884.500,      894.500,      907.272705078125,  914.500,      924.500},
      {1474.50,      1484.50,      1494.50,      1504.50,      0.0,      1524.50},
      {2074.50,      2084.50,      2094.50,      2104.50,      2114.50,      2124.50}
    };

    if (result.nline != RESULT_NUMBER_LINE ||
	result.nsample != RESULT_NUMBER_SAMPLE ||
	result_mask.nline != RESULT_NUMBER_LINE ||
	result_mask.nsample != RESULT_NUMBER_SAMPLE) {
      fprintf(stderr,"result.nline = %d (expected %d)\n",
	      result.nline, RESULT_NUMBER_LINE);
      fprintf(stderr,"result.nsample = %d (expected %d)\n",
	      result.nsample, RESULT_NUMBER_SAMPLE);
      fprintf(stderr,"result_mask.nline = %d (expected %d)\n",
	      result_mask.nline, RESULT_NUMBER_LINE);
      fprintf(stderr,"result_mask.nsample = %d (expected %d)\n",
	      result_mask.nsample, RESULT_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }

    for (iline = 0 ; iline < result_mask.nline ; iline++) {
      for (isample = 0 ; isample < result_mask.nsample ; isample++) {
	if (result_mask.data.u8[iline][isample] !=
	    result_mask_expect[iline][isample]) {
	  fprintf(stderr,"result_mask.data.u8[%d][%d] = %d (expected %d)\n",
		  iline, isample, 
		  result_mask.data.u8[iline][isample],
		  result_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    for (iline = 0 ; iline < result.nline ; iline++) {
      for (isample = 0 ; isample < result.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(result.data.f[iline][isample],
			    result_expect[iline][isample])) {
	  fprintf(stderr,"result.data.u8[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, 
		  result.data.f[iline][isample],
		  result_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }
    
    if (error) {
      fprintf(stderr,"result_mask:\n");
      for (iline = 0 ; iline < result_mask.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < result_mask.nsample ; isample++) {
	  fprintf(stderr,"%d, ",result_mask.data.u8[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
      fprintf(stderr,"result:\n");
      for (iline = 0 ; iline < result.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < result.nsample ; isample++) {
	  fprintf(stderr,"%20.20g, ",result.data.f[iline][isample]);
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

  status = MtkDownsample(NULL, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source.nline = 0;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nline = NUMBER_LINE;

  source.nsample = 0;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nsample = NUMBER_SAMPLE;

  source.datatype++;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
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

  status = MtkDownsample(&source, 
			 NULL, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source_mask.nline++;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nline--;

  source_mask.nsample++;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nsample--;

  source_mask.datatype++;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   Size_factor < 1                                                  */
  /*   Source->nline % Size_factor != 0                                 */
  /*   Source->nsample % Size_factor != 0                               */
  /* ------------------------------------------------------------------ */

  size_factor = 0;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  size_factor = 10;

  size_factor = 60;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  size_factor = 10;

  size_factor = 40;
  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
			 &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  size_factor = 10;

  /* ------------------------------------------------------------------ */
  /* Argument check: Result == NULL                                     */
  /* ------------------------------------------------------------------ */

  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 NULL,
			 &result_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Result_mask == NULL                                */
  /* ------------------------------------------------------------------ */

  status = MtkDownsample(&source, 
			 &source_mask, 
			 size_factor,
			 &result,
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
