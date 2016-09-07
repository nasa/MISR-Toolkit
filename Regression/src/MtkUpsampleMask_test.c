/*===========================================================================
=                                                                           =
=                           MtkUpsampleMask_test                            =
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

#define NUMBER_LINE 2
#define NUMBER_SAMPLE 3

#define RESULT_NUMBER_LINE 4
#define RESULT_NUMBER_SAMPLE 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer source_mask = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer result_mask = MTKT_DATABUFFER_INIT;
  int size_factor = 2;
  int iline, isample;
  
  MTK_PRINT_STATUS(cn,"Testing MtkUpsampleMask");
  fprintf(stderr,"\n");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

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
      source_mask.data.u8[iline][isample] = 1;
    }
  }
  source_mask.data.u8[1][0] = 0;

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   &result_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkUpsampleMask(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  {
    uint8 result_mask_expect[RESULT_NUMBER_LINE][RESULT_NUMBER_SAMPLE] = {
      { 1, 1, 1, 1, 1, 1 },
      { 1, 1, 1, 1, 1, 1 },
      { 0, 0, 1, 1, 1, 1 },
      { 0, 0, 1, 1, 1, 1 }
    };

    if (result_mask.nline != RESULT_NUMBER_LINE ||
	result_mask.nsample != RESULT_NUMBER_SAMPLE) {
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

    if (error) {
      fprintf(stderr,"result_mask:\n");
      for (iline = 0 ; iline < result_mask.nline ; iline++) {
	fprintf(stderr,"{ ");
	for (isample = 0 ; isample < result_mask.nsample ; isample++) {
	  fprintf(stderr,"%d, ",result_mask.data.u8[iline][isample]);
	}
	fprintf(stderr,"},\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Source_mask == NULL                                */
  /*                 Source_mask->nline < 1                             */
  /*                 Source_mask->nsample < 1                           */
  /*                 Source_mask->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  status = MtkUpsampleMask(NULL, 
			   size_factor,
			   &result_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source_mask.nline = 0;
  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(2)\n");
    error = MTK_TRUE;
  }
  source_mask.nline = NUMBER_LINE;

  source_mask.nsample = 0;
  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(3)\n");
    error = MTK_TRUE;
  }
  source_mask.nsample = NUMBER_SAMPLE;

  source_mask.datatype++;
  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(4)\n");
    error = MTK_TRUE;
  }
  source_mask.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   Size_factor < 1                                                  */
  /* ------------------------------------------------------------------ */

  size_factor = 0;
  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   &result_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(5)\n");
    error = MTK_TRUE;
  }
  size_factor = 2;

  /* ------------------------------------------------------------------ */
  /* Argument check: Result_mask == NULL                                */
  /* ------------------------------------------------------------------ */

  status = MtkUpsampleMask(&source_mask, 
			   size_factor,
			   NULL);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(6)\n");
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
