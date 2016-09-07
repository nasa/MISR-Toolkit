/*===========================================================================
=                                                                           =
=                           MtkSmoothData_test                              =
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

#define MTKm_CMP_NE_FLT(x,y) (fabs((x)-(y)) > FLT_EPSILON * 10 * fabs(x))

#define NUMBER_LINE 4
#define NUMBER_SAMPLE 6

int main () {
  MTKt_status status;           /* Return status */
  MTKt_boolean error = MTK_FALSE; /* Test status */
  int cn = 0;
  MTKt_DataBuffer data = MTKT_DATABUFFER_INIT;
  MTKt_DataBuffer valid_mask = MTKT_DATABUFFER_INIT;
  int width_line = 3;
  int width_sample = 5;
  MTKt_DataBuffer data_smoothed = MTKT_DATABUFFER_INIT;
  int iline; 			
  int isample;
  
  MTK_PRINT_STATUS(cn,"Testing MtkSmoothData");

  /* ------------------------------------------------------------------ */
  /* Initialize test input.                                             */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_float,
				 &data);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  status = MtkDataBufferAllocate(NUMBER_LINE,
				 NUMBER_SAMPLE,
				 MTKe_uint8,
				 &valid_mask);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkDataBufferAllocate(1)\n");
    error = MTK_TRUE;
  }

  for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
    for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
      data.data.f[iline][isample] = cos(iline) + sin(isample);
      valid_mask.data.u8[iline][isample] = 1;
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 1                                                      */
  /* ------------------------------------------------------------------ */

  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkSmoothData(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
   
  {
    float data_smoothed_expect[NUMBER_LINE][NUMBER_SAMPLE] = {
      {1.1414962493914384556, 1.5223521971009816856, 1.3012113674826761844, 0.91729565793898615311, 0.43553153907640168585, 0.12275832812091826141},
      {0.79110569853817125363, 0.95450713389282904053, 0.60173567465076926997, 0.40995081971814151256, 0.1234128869138712975, -0.2575390460360399425},
      {0.010527542251969057574, 0.23255263964965353085, -0.061595157549379303541, -0.25338001248200686666, -0.59854160732930383748, -1.0381172023222418055},
      {-0.64178757637197436647, -0.19202873838342018886, -0.34426667772271507539, -0.72818238726640482916, -1.2788493964080001053, -1.6605254976424939084}
    };

    for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
      for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
	if (MTKm_CMP_NE_FLT(data_smoothed.data.f[iline][isample],
			    data_smoothed_expect[iline][isample])) {
	  printf("data_smoothed.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		 iline, isample, 
		 data_smoothed.data.f[iline][isample],
		 data_smoothed_expect[iline][isample]);
	  printf("Unexpected result(test1).\n");
	  error = MTK_TRUE;

	}
      } 
    }
  }

  /* ------------------------------------------------------------------ */
  /* Normal test 2                                                      */
  /* ------------------------------------------------------------------ */

  valid_mask.data.u8[1][2] = 0;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkSmoothData(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Check result                                                       */
  /* ------------------------------------------------------------------ */
   
  {
    float data_smoothed_expect[NUMBER_LINE][NUMBER_SAMPLE] = {
      {1.1115229338785170832, 1.5484769472419197545, 1.3318578804248000225, 0.89673034296338915983, 0.35510472387628505553, 0.12275832812091826141 },
      {0.73048587008312582114, 0.94998537109164338244,                    0, 0.35873899180042040369, 0.012339558771630738943, -0.2575390460360399425 },
      {-0.11385556236409519193, 0.16426760068744927779, -0.12535843371039806504, -0.36835511656074670928, -0.7733782116325631506, -1.0381172023222418055 },
      {-0.64178757637197436647, -0.19202873838342018886, -0.34426667772271507539, -0.72818238726640482916, -1.2788493964080001053, -1.6605254976424939084 }
    };

    for (iline = 0 ; iline < NUMBER_LINE ; iline++) {
      for (isample = 0 ; isample < NUMBER_SAMPLE ; isample++) {
	if (MTKm_CMP_NE_FLT(data_smoothed.data.f[iline][isample],
			    data_smoothed_expect[iline][isample])) {
	  printf("data_smoothed.data.f[%d][%d] = %20.20g (expected %20.20g)\n",
		 iline, isample, 
		 data_smoothed.data.f[iline][isample],
		 data_smoothed_expect[iline][isample]);
	  printf("Unexpected result(test1).\n");
	  error = MTK_TRUE;

	}
      } 
    }
  }

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferFree(&data_smoothed);
  if (status != MTK_SUCCESS) {
    printf("Trouble with MtkDataBufferFree(1)\n");
    error = MTK_TRUE;
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Data == NULL                                       */
  /*                 Data->nline < 1                                    */
  /*                 Data->nsample < 1                                  */
  /*                 Data->datatype != MTKe_float                       */
  /* ------------------------------------------------------------------ */

  status = MtkSmoothData(NULL, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(1a)\n");
    error = MTK_TRUE;
  }

  data.nline = 0;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(1b)\n");
    error = MTK_TRUE;
  }
  data.nline = NUMBER_LINE;

  data.nsample = 0;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(1c)\n");
    error = MTK_TRUE;
  }
  data.nsample = NUMBER_SAMPLE;

  data.datatype++;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(1d)\n");
    error = MTK_TRUE;
  }
  data.datatype--;

  /* ------------------------------------------------------------------ */
  /* Argument check: Valid_mask == NULL                                 */
  /*                 Valid_mask->nline != Data->nline                   */
  /*                 Valid_mask->nsample != Data->nsample               */
  /*                 Valid_mask->datatype != MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  status = MtkSmoothData(&data, NULL, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(2a)\n");
    error = MTK_TRUE;
  }

  valid_mask.nline++;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(2b)\n");
    error = MTK_TRUE;
  }
  valid_mask.nline--;

  valid_mask.nsample++;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(2c)\n");
    error = MTK_TRUE;
  }
  valid_mask.nsample--;

  valid_mask.datatype = MTKe_int8;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(2d)\n");
    error = MTK_TRUE;
  }
  valid_mask.datatype = MTKe_uint8;

  /* ------------------------------------------------------------------ */
  /* Argument check: Width_line < 1                                     */
  /*                 Width_line > Data->nline                           */
  /*                 Width_line is not odd                              */
  /* ------------------------------------------------------------------ */

  width_line = -1;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(3a)\n");
    error = MTK_TRUE;
  }
  width_line = 1;

  width_line = NUMBER_LINE + 1;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(3b)\n");
    error = MTK_TRUE;
  }
  width_line = NUMBER_LINE - 1;

  width_line = 2;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(3c)\n");
    error = MTK_TRUE;
  }
  width_line = 3;

  /* ------------------------------------------------------------------ */
  /* Argument check: Width_sample < 1                                   */
  /*                 Width_sample > Data->nsample                       */
  /*                 Width_sample is not odd                            */
  /* ------------------------------------------------------------------ */

  width_sample = -1;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(4a)\n");
    error = MTK_TRUE;
  }
  width_sample = 1;

  width_sample = NUMBER_SAMPLE + 1;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(4b)\n");
    error = MTK_TRUE;
  }
  width_sample = NUMBER_SAMPLE - 1;

  width_sample = 2;
  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 &data_smoothed);
  if (status != MTK_OUTBOUNDS) {
    printf("Unexpected status(4c)\n");
    error = MTK_TRUE;
  }
  width_sample = 3;

  /* ------------------------------------------------------------------ */
  /* Argument check: Data_smoothed = NULL                               */
  /* ------------------------------------------------------------------ */

  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, 
			 NULL);
  if (status != MTK_NULLPTR) {
    printf("Unexpected status(5)\n");
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
