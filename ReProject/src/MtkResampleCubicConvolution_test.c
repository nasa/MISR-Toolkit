/*===========================================================================
=                                                                           =
=                           MtkResampleCubicConvolution_test                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

#define MTKm_CMP_NE_FLT(x,y) (fabs((x)-(y)) > FLT_EPSILON * 10 * fabs(x))

#define NUMBER_LINE 40
#define NUMBER_SAMPLE 60
#define RESAMPLED_NUMBER_LINE 4
#define RESAMPLED_NUMBER_SAMPLE 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer source = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer source_mask = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer line = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer sample = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer resampled = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer resampled_mask = MTKT_DATABUFFER_INIT;
  int iline, isample;
  float a = -0.5;
  
  MTK_PRINT_STATUS(cn,"Testing MtkResampleCubicConvolution");

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
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(2)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(RESAMPLED_NUMBER_LINE,
				 RESAMPLED_NUMBER_SAMPLE,
				 MTKe_float,
				 &line);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(3)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(RESAMPLED_NUMBER_LINE,
				 RESAMPLED_NUMBER_SAMPLE,
				 MTKe_float,
				 &sample);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkDataBufferAllocate(4)\n");
    error = MTK_TRUE;
  }


  for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
      source.data.f[iline][isample] = iline * 0.04 + isample * 0.06;
      source_mask.data.u8[iline][isample] = 1;
    }
  }
  
  for (iline = 0 ; iline < RESAMPLED_NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < RESAMPLED_NUMBER_SAMPLE ; isample++) {
      line.data.f[iline][isample] = 10.0 * iline + 4.1;
      sample.data.f[iline][isample] = 10.0 * isample + 2.1;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkResampleCubicConvolution(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  { 
    float resampled_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {0.290000, 0.890000, 1.49000, 2.09000, 2.69000, 3.29000},
      {0.690000, 1.29000,  1.89000, 2.49000, 3.09000, 3.69000},
      {1.09000,  1.69000,  2.29000, 2.89000, 3.49000, 4.09000},
      {1.49000,  2.09000,  2.69000, 3.29000, 3.89000, 4.49000}
    };

    uint8 resampled_mask_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {1,1,1,1,1,1},
      {1,1,1,1,1,1},
      {1,1,1,1,1,1},
      {1,1,1,1,1,1}
    };

    if (resampled.nline != RESAMPLED_NUMBER_LINE ||
	resampled.nsample != RESAMPLED_NUMBER_SAMPLE ||
	resampled_mask.nline != RESAMPLED_NUMBER_LINE ||
	resampled_mask.nsample != RESAMPLED_NUMBER_SAMPLE) {
      fprintf(stderr,"resampled.nline = %d (expected %d)\n",
	      resampled.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled.nsample = %d (expected %d)\n",
	      resampled.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"resampled_mask.nline = %d (expected %d)\n",
	      resampled_mask.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled_mask.nsample = %d (expected %d)\n",
	      resampled_mask.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test1).\n");
      error = MTK_TRUE;
    }
    
    for (iline = 0 ; iline < resampled.nline ; iline++) {
      for (isample = 0 ; isample < resampled.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(resampled.data.f[iline][isample],
			   resampled_expect[iline][isample]) ||
	    resampled_mask.data.u8[iline][isample] != resampled_mask_expect[iline][isample]) {
	  fprintf(stderr,"resampled.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, 
		  resampled.data.f[iline][isample],
		  resampled_expect[iline][isample]);
	  fprintf(stderr,"resampled_mask.data.u8[%d][%d] = %d (expected %d)\n",
		  iline, isample, 
		  resampled_mask.data.u8[iline][isample],
		  resampled_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test1).\n");
	  error = MTK_TRUE;
	}
      }
    }

    if (error) {
      printf("\nline:\n");
      for (iline = 0 ; iline < line.nline; iline++) {
	for (isample = 0 ; isample < line.nsample; isample++) {
	  printf("%6.2g ",line.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nsample:\n");
      for (iline = 0 ; iline < sample.nline; iline++) {
	for (isample = 0 ; isample < sample.nsample; isample++) {
	  printf("%6.2g ",sample.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled_mask:\n");
      for (iline = 0 ; iline < resampled_mask.nline; iline++) {
	for (isample = 0 ; isample < resampled_mask.nsample; isample++) {
	  printf("%6d ",resampled_mask.data.u8[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled:\n");
      for (iline = 0 ; iline < resampled.nline; iline++) {
	for (isample = 0 ; isample < resampled.nsample; isample++) {
	  printf("%20.13f ",resampled.data.f[iline][isample]);
	}
	printf("\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  line.data.f[2][3] = -10.0;
  sample.data.f[2][5] = 99.0;
  line.data.f[1][2] = NUMBER_LINE-1.0;
  sample.data.f[1][2] = NUMBER_SAMPLE-1.0;

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkResampleCubicConvolution(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  { 
    float resampled_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {0.290000, 0.890000, 1.49000, 2.09000, 2.69000, 3.29000},
      {0.690000, 1.29000,  5.1,     2.49000, 3.09000, 3.69000},
      {1.09000,  1.69000,  2.29000, 0.0, 3.49000, 0.0},
      {1.49000,  2.09000,  2.69000, 3.29000, 3.89000, 4.49000}
    };

    uint8 resampled_mask_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {1,1,1,1,1,1},
      {1,1,1,1,1,1},
      {1,1,1,0,1,0},
      {1,1,1,1,1,1}
    };

    if (resampled.nline != RESAMPLED_NUMBER_LINE ||
	resampled.nsample != RESAMPLED_NUMBER_SAMPLE ||
	resampled_mask.nline != RESAMPLED_NUMBER_LINE ||
	resampled_mask.nsample != RESAMPLED_NUMBER_SAMPLE) {
      fprintf(stderr,"resampled.nline = %d (expected %d)\n",
	      resampled.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled.nsample = %d (expected %d)\n",
	      resampled.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"resampled_mask.nline = %d (expected %d)\n",
	      resampled_mask.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled_mask.nsample = %d (expected %d)\n",
	      resampled_mask.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test2).\n");
      error = MTK_TRUE;
    }
    
    for (iline = 0 ; iline < resampled.nline ; iline++) {
      for (isample = 0 ; isample < resampled.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(resampled.data.f[iline][isample],
			   resampled_expect[iline][isample]) ||
	    resampled_mask.data.u8[iline][isample] != resampled_mask_expect[iline][isample]) {
	  fprintf(stderr,"resampled.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, 
		  resampled.data.f[iline][isample],
		  resampled_expect[iline][isample]);
	  fprintf(stderr,"resampled_mask.data.u8[%d][%d] = %d (expected %d)\n",
		  iline, isample, 
		  resampled_mask.data.u8[iline][isample],
		  resampled_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test2).\n");
	  error = MTK_TRUE;
	}
      }
    }

    if (error) {
      printf("\nline:\n");
      for (iline = 0 ; iline < line.nline; iline++) {
	for (isample = 0 ; isample < line.nsample; isample++) {
	  printf("%6.2g ",line.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nsample:\n");
      for (iline = 0 ; iline < sample.nline; iline++) {
	for (isample = 0 ; isample < sample.nsample; isample++) {
	  printf("%6.2g ",sample.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled_mask:\n");
      for (iline = 0 ; iline < resampled_mask.nline; iline++) {
	for (isample = 0 ; isample < resampled_mask.nsample; isample++) {
	  printf("%6d ",resampled_mask.data.u8[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled:\n");
      for (iline = 0 ; iline < resampled.nline; iline++) {
	for (isample = 0 ; isample < resampled.nsample; isample++) {
	  printf("%20.13f ",resampled.data.f[iline][isample]);
	}
	printf("\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 3                                                      */
  /* ------------------------------------------------------------------ */

  for (iline = 0 ; iline < RESAMPLED_NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < RESAMPLED_NUMBER_SAMPLE ; isample++) {
      line.data.f[iline][isample] = 10.0 * iline + 4.1;
      sample.data.f[iline][isample] = 10.0 * isample + 2.1;
    }
  }

  for (iline = 1 ; iline < NUMBER_LINE ; iline+=2) {
    for (isample = 1 ; isample < NUMBER_SAMPLE ; isample+=2) {
      source_mask.data.u8[iline][isample] = 0;
    }
  }
  source_mask.data.u8[14][12] = 0;

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_SUCCESS) {
    fprintf(stderr,"Trouble with MtkResampleCubicConvolution(3)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */

  { 
    float resampled_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {0.2896948, 0.8896948, 1.4896948, 2.0896948, 2.6896948, 3.2896948},
      {0.6896948, 0.0,       1.8896948, 2.4896948, 3.0896948, 3.6896948},
      {1.0896948, 1.6896948, 2.2896948, 2.8896948, 3.4896948, 4.0896948},
      {1.4896948, 2.0896948, 2.6896948, 3.2896948, 3.8896948, 4.4896948}
    };

    uint8 resampled_mask_expect[RESAMPLED_NUMBER_LINE][RESAMPLED_NUMBER_SAMPLE] = {
      {1,1,1,1,1,1},
      {1,0,1,1,1,1},
      {1,1,1,1,1,1},
      {1,1,1,1,1,1}
    };

    if (resampled.nline != RESAMPLED_NUMBER_LINE ||
	resampled.nsample != RESAMPLED_NUMBER_SAMPLE ||
	resampled_mask.nline != RESAMPLED_NUMBER_LINE ||
	resampled_mask.nsample != RESAMPLED_NUMBER_SAMPLE) {
      fprintf(stderr,"resampled.nline = %d (expected %d)\n",
	      resampled.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled.nsample = %d (expected %d)\n",
	      resampled.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"resampled_mask.nline = %d (expected %d)\n",
	      resampled_mask.nline, RESAMPLED_NUMBER_LINE);
      fprintf(stderr,"resampled_mask.nsample = %d (expected %d)\n",
	      resampled_mask.nsample, RESAMPLED_NUMBER_SAMPLE);
      fprintf(stderr,"Unexpected result(test3).\n");
      error = MTK_TRUE;
    }
    
    for (iline = 0 ; iline < resampled.nline ; iline++) {
      for (isample = 0 ; isample < resampled.nsample ; isample++) {
	if (MTKm_CMP_NE_FLT(resampled.data.f[iline][isample],
			   resampled_expect[iline][isample]) ||
	    resampled_mask.data.u8[iline][isample] != resampled_mask_expect[iline][isample]) {
	  fprintf(stderr,"resampled.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		  iline, isample, 
		  resampled.data.f[iline][isample],
		  resampled_expect[iline][isample]);
	  fprintf(stderr,"resampled_mask.data.u8[%d][%d] = %d (expected %d)\n",
		  iline, isample, 
		  resampled_mask.data.u8[iline][isample],
		  resampled_mask_expect[iline][isample]);
	  fprintf(stderr,"Unexpected result(test3).\n");
	  error = MTK_TRUE;
	}
      }
    }

    if (error) {
      printf("\nline:\n");
      for (iline = 0 ; iline < line.nline; iline++) {
	for (isample = 0 ; isample < line.nsample; isample++) {
	  printf("%6.2g ",line.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nsample:\n");
      for (iline = 0 ; iline < sample.nline; iline++) {
	for (isample = 0 ; isample < sample.nsample; isample++) {
	  printf("%6.2g ",sample.data.f[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled_mask:\n");
      for (iline = 0 ; iline < resampled_mask.nline; iline++) {
	for (isample = 0 ; isample < resampled_mask.nsample; isample++) {
	  printf("%6d ",resampled_mask.data.u8[iline][isample]);
	}
	printf("\n");
      }
      printf("\nresampled:\n");
      for (iline = 0 ; iline < resampled.nline; iline++) {
	for (isample = 0 ; isample < resampled.nsample; isample++) {
	  printf("%20.13f ",resampled.data.f[iline][isample]);
	}
	printf("\n");
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Source == NULL                                     */
  /*                 Source->nline < 1                                  */
  /*                 Source->nsample < 1                                */
  /*                 Source->datatype == MTKe_float                     */
  /* ------------------------------------------------------------------ */

  fprintf(stderr,"\n");
  status = MtkResampleCubicConvolution(NULL, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source.nline = 0;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nline = NUMBER_LINE;

  source.nsample = 0;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source.nsample = NUMBER_SAMPLE;

  source.datatype++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
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

  status = MtkResampleCubicConvolution(&source, 
				       NULL, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  source_mask.nline++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nline--;

  source_mask.nsample++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.nsample--;

  source_mask.datatype++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  source_mask.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Line == NULL                                       */
  /*                 Line->nline < 1                                    */
  /*                 Line->nsample < 1                                  */
  /*                 Line->datatype != MTKe_float                       */
  /* ------------------------------------------------------------------ */

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       NULL, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  line.nline = 0;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  line.nline = RESAMPLED_NUMBER_LINE;

  line.nsample = 0;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  line.nsample = RESAMPLED_NUMBER_SAMPLE;

  line.datatype++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  line.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Sample == NULL                                     */
  /*                 Sample->nline != Line->nline                       */
  /*                 Sample->nsample != Line->nsample                   */
  /*                 Sample->datatype != MTKe_float                     */
  /* ------------------------------------------------------------------ */

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       NULL, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  sample.nline++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  sample.nline--;

  sample.nsample++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  sample.nsample--;

  sample.datatype++;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  sample.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: A > 0.0                                            */
  /*                 A < -1.0                                           */
  /* ------------------------------------------------------------------ */

  a = 0.00001;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  a = 0.0;

  a = -1.0001;
  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
				       &resampled_mask);
  if (status != MTK_OUTBOUNDS) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }
  a = -1.0;

  /* ------------------------------------------------------------------ */
  /* Argument check: Resampled == NULL                                  */
  /* ------------------------------------------------------------------ */

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       NULL,
				       &resampled_mask);
  if (status != MTK_NULLPTR) {
    fprintf(stderr,"Unexpected status(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Resampled_mask == NULL                             */
  /* ------------------------------------------------------------------ */

  status = MtkResampleCubicConvolution(&source, 
				       &source_mask, 
				       &line, 
				       &sample, 
				       a,
				       &resampled,
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
