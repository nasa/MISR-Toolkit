/*===========================================================================
=                                                                           =
=                          MtkRegressionCoeffCalc                           =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrRegression.h"
#include "MisrUtil.h"
#include <stdlib.h>
#include <math.h>

/** \brief  Calculate linear regression coefficients for translating values in data buffer 1 to corresponding values in data buffer 2.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example we determine coefficients to map values in data1 with mask valid_mask1 to data2 having mask valid_mask2 and uncertainty data2_sigma. Linear regression will use pixel neighborhood size specified by regression_size_factor and output the results to regression_coeff with regression_coeff_map_info.
 *
 *  \code
 *  status = MtkRegressionCoeffCalc(&data1, &valid_mask1, &data2, &data2_sigma, &valid_mask2, &map_info, regression_size_factor, &regression_coeff, &regression_coeff_map_info); 
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkRegressionCoeffCalc(
  const MTKt_DataBuffer *Data1, /**< [IN] Data buffer 1 */
  const MTKt_DataBuffer *Valid_mask1, /**< [IN] Valid mask for data buffer 1 */
  const MTKt_DataBuffer *Data2, /**< [IN] Data buffer 2 */
  const MTKt_DataBuffer *Data2_sigma, /**< [IN] Uncertainty for data buffer 2 */
  const MTKt_DataBuffer *Valid_mask2, /**< [IN] Valid mask for data buffer 2 */
  const MTKt_MapInfo *Map_info, /**< [IN] Map info for input data. */
  int Regression_size_factor, /**< [IN] Number of pixels to aggregate along each axis when generating regression coefficients. */
  MTKt_RegressionCoeff *Regression_coeff, /**< [OUT] Regression coefficients. */
  MTKt_MapInfo *Regression_coeff_map_info /**< [OUT] Map info for regression coefficients */
) 
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_RegressionCoeff regression_coeff_tmp = MTKT_REGRESSION_COEFF_INIT;
  MTKt_MapInfo regression_coeff_map_info_tmp = MTKT_MAPINFO_INIT;
  int iline;
  int isample;
  int number_line_out; 		/* Number of lines in output grid.*/
  int number_sample_out; 	/* Number of samples in output grid.*/
  int number_line_in; 		/* Number of lines in input grid. */
  int number_sample_in; 	/* Number of samples in input grid. */
  double *data1 = NULL;		/* Temporary buffer to contain data values. */
  double *data2 = NULL;		/* Temporary buffer to contain data values. */
  double *data2_sigma = NULL;  /* Temporary buffer to contain uncertainty. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Data1 == NULL                                      */
  /*                 Data1->nline < 1                                   */
  /*                 Data1->nsample < 1                                 */
  /*                 Data1->datatype = MTKe_float                       */
  /* ------------------------------------------------------------------ */

  if (Data1 == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Data1 == NULL");
  }
  if (Data1->nline < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data1->nline < 1");
  }
  if (Data1->nsample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data1->nsample < 1");
  }
  if (Data1->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data1->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Get size of input data.                                            */
  /* ------------------------------------------------------------------ */
  
  number_line_in = Data1->nline;
  number_sample_in = Data1->nsample;

  /* ------------------------------------------------------------------ */
  /* Argument check: Valid_mask1 == NULL                                */
  /*                 Valid_mask1->nline != Data1->nline                 */
  /*                 Valid_mask1->nsample != Data1->nsample             */
  /*                 Valid_mask1->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  if (Valid_mask1 == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Valid_mask1 == NULL");
  }
  if (Valid_mask1->nline != number_line_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask1->nline != Data1->nline");
  }
  if (Valid_mask1->nsample != number_sample_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask1->nsample != Data1->nsample");
  }
  if (Valid_mask1->datatype != MTKe_uint8) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask1->datatype != MTKe_uint8");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Data2 == NULL                                      */
  /*                 Data2->nline != Data1->nline                       */
  /*                 Data2->nsample != Data1->nsample                   */
  /*                 Data2->datatype = MTKe_float                      */
  /* ------------------------------------------------------------------ */

  if (Data2 == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Data2 == NULL");
  }
  if (Data2->nline != number_line_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2->nline != Data1->nline");
  }
  if (Data2->nsample != number_sample_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2->nsample != Data1->nsample");
  }
  if (Data2->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Data2_sigma == NULL                                */
  /*                 Data2_sigma->nline != Data1->nline                 */
  /*                 Data2_sigma->nsample != Data1->nsample             */
  /*                 Data2_sigma->datatype = MTKe_float                */
  /* ------------------------------------------------------------------ */

  if (Data2_sigma == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Data2_sigma == NULL");
  }
  if (Data2_sigma->nline != number_line_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2_sigma->nline != Data1->nline");
  }
  if (Data2_sigma->nsample != number_sample_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2_sigma->nsample != Data1->nsample");
  }
  if (Data2_sigma->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data2_sigma->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Valid_mask2 == NULL                                */
  /*                 Valid_mask2->nline != Data1->nline                 */
  /*                 Valid_mask2->nsample != Data1->nsample             */
  /*                 Valid_mask2->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  if (Valid_mask2 == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Valid_mask2 == NULL");
  }
  if (Valid_mask2->nline != number_line_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask2->nline != Data1->nline");
  }
  if (Valid_mask2->nsample != number_sample_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask2->nsample != Data1->nsample");
  }
  if (Valid_mask2->datatype != MTKe_uint8) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Valid_mask2->datatype != MTKe_uint8");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Map_info == NULL                                   */
  /*                 Map_info->nline != Data1->nline                    */
  /*                 Map_info->nsample != Data1->nsample                */
  /* ------------------------------------------------------------------ */

  if (Map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Map_info == NULL");
  }
  if (Map_info->nline != number_line_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Map_info->nline != Data1->nline");
  }
  if (Map_info->nsample != number_sample_in) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Map_info->nsample != Data1->nsample");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   Data1->nline % Regression_size_factor != 0                       */
  /*   Data1->nsample in % Regression_size_factor != 0                  */
  /* ------------------------------------------------------------------ */
  
  if (number_line_in % Regression_size_factor != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data1->nline % Regression_size_factor != 0");
  }
  if (number_sample_in % Regression_size_factor != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Data1->nsample % Regression_size_factor != 0");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff == NULL                           */
  /* ------------------------------------------------------------------ */

  if (Regression_coeff == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regression_coeff == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff_map_info == NULL                  */
  /* ------------------------------------------------------------------ */

  if (Regression_coeff_map_info == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regression_coeff_map_info == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Determine size of output array.                                    */
  /* ------------------------------------------------------------------ */

  number_line_out = number_line_in / Regression_size_factor;
  number_sample_out = number_sample_in / Regression_size_factor; 

  /* ------------------------------------------------------------------ */
  /* Allocate memory for regression coefficients                        */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffAllocate(number_line_out, number_sample_out, 
				      &regression_coeff_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Allocate memory.                                                   */
  /* ------------------------------------------------------------------ */

  data1 = (double *)calloc(Regression_size_factor*Regression_size_factor,
			   sizeof(double));
  if (data1 == NULL) {
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
  }
  data2 = (double *)calloc(Regression_size_factor*Regression_size_factor,
			   sizeof(double));
  if (data2 == NULL) {
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
  }
  data2_sigma = (double *)calloc(Regression_size_factor*Regression_size_factor,
				 sizeof(double));
  if (data2_sigma == NULL) {
    MTK_ERR_CODE_JUMP(MTK_CALLOC_FAILED);
  }

  /* ------------------------------------------------------------------ */
  /* For each location in regression coefficient grid...                */
  /* ------------------------------------------------------------------ */

  for (iline = 0; iline < number_line_out ; iline++) {
    for (isample = 0; isample < number_sample_out; isample++) {
      int rline_start, rsample_start;
      int rline_end, rsample_end;
      int rline, rsample;
      int count;
      double intercept, slope, correlation;

      rline_start = iline * Regression_size_factor;
      rsample_start = isample * Regression_size_factor;
      rline_end = rline_start + Regression_size_factor;
      rsample_end = rsample_start + Regression_size_factor;
      count = 0;
      
  /* ------------------------------------------------------------------ */
  /* Calculate linear regression at this location.                      */
  /* If the linear regression cannot be calculated due to a             */
  /* divide-by-zero condition, then skip this location.                 */
  /* ------------------------------------------------------------------ */

      for (rline = rline_start ; rline < rline_end ; rline++) {
	for (rsample = rsample_start ; rsample < rsample_end ; rsample++) {
	  if (Valid_mask1->data.u8[rline][rsample] &&
	      Valid_mask2->data.u8[rline][rsample]) {
	    data1[count] = Data1->data.f[rline][rsample];
	    data2[count] = Data2->data.f[rline][rsample];
	    data2_sigma[count] = Data2_sigma->data.f[rline][rsample];
	    count++;
	  }
	}
      }

      regression_coeff_tmp.valid_mask.data.u8[iline][isample] = 0;

      if (count > 0) {
	status = MtkLinearRegressionCalc(count, data1, data2, data2_sigma, 
					 &intercept, &slope, &correlation);
	if (status == MTK_DIV_BY_ZERO) {
	  continue;
	} else {
	  MTK_ERR_COND_JUMP(status);
	}

	regression_coeff_tmp.intercept.data.f[iline][isample] = intercept;
	regression_coeff_tmp.slope.data.f[iline][isample] = slope;
	regression_coeff_tmp.correlation.data.f[iline][isample] = correlation;

	regression_coeff_tmp.valid_mask.data.u8[iline][isample] = 1;
      }
      
  /* ------------------------------------------------------------------ */
  /* End loop for each location in regression coefficient grid.         */
  /* ------------------------------------------------------------------ */

    }
  }

  /* ------------------------------------------------------------------ */
  /* Set map info for regression coefficients.                          */
  /* ------------------------------------------------------------------ */

  status = MtkChangeMapResolution(Map_info,
				  Map_info->resolution * Regression_size_factor,
				  &regression_coeff_map_info_tmp);
  MTK_ERR_COND_JUMP(status);
  
  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Regression_coeff = regression_coeff_tmp;
  *Regression_coeff_map_info = regression_coeff_map_info_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  if (data1 != NULL) {
    free(data1);
  }
  if (data2 != NULL) {
    free(data2);
  }
  if (data2_sigma != NULL) {
    free(data2_sigma);
  }

  MtkRegressionCoeffFree(&regression_coeff_tmp);
  return status_code;
}
