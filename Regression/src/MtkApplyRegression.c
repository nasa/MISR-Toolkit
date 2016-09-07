/*===========================================================================
=                                                                           =
=                          MtkApplyRegression                               =
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

/** \brief Apply regression to given data.  Uses MtkResampleCubicConvolution to resample regression coefficients to resolution of the input data.  Output data is the same size and resolution as the input data.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example: 
 *  In this example we apply a regression based on the supplied source, source_mask, and source_map_info using regression_coeff and regression_coeff_map_info. The result is regressed and regressed_mask.
 *
 *  \code 
 *  status = MtkApplyRegression(&source, &source_mask, &source_map_info, &regression_coeff, &regression_coeff_map_info, &regressed, &regressed_mask);
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkApplyRegression(
  const MTKt_DataBuffer *Source, /**< [IN] Input data. (float) */
  const MTKt_DataBuffer *Source_mask, /**< [IN] Valid mask for input data. (uint8) */
  const MTKt_MapInfo    *Source_map_info, /**< [IN] Map info for input data */
  const MTKt_RegressionCoeff *Regression_coeff, /**< [IN] Regression coefficients */
  const MTKt_MapInfo *Regression_coeff_map_info, /**< [IN] Map info for regression coefficients */
  MTKt_DataBuffer *Regressed,  /**< [OUT] Output data.  (float) */
  MTKt_DataBuffer *Regressed_mask /**< [OUT] Valid mask for output data. (uint8) */
)
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_DataBuffer regressed_tmp = MTKT_DATABUFFER_INIT;
				/* Output data. */
  MTKt_DataBuffer regressed_mask_tmp = MTKT_DATABUFFER_INIT;
				/* Output data mask */
  MTKt_RegressionCoeff regression_coeff_resampled = MTKT_REGRESSION_COEFF_INIT;
				/* Regression coefficients resampled to 
				   resolution of source data.  */
  int iline;  			/* Loop iterator. */
  int isample; 			/* Loop iterator. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Source == NULL                                     */
  /*                 Source->nline < 1                                  */
  /*                 Source->nsample < 1                                */
  /*                 Source->datatype = MTKe_float                      */
  /* ------------------------------------------------------------------ */

  if (Source == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Source == NULL");
  }
  if (Source->nline < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source->nline < 1");
  }
  if (Source->nsample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source->nsample < 1");
  }
  if (Source->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Source_mask == NULL                                */
  /*                 Source_mask->nline != Source->nline                */
  /*                 Source_mask->nsample != Source->nsample            */
  /*                 Source_mask->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  if (Source_mask == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Source_mask == NULL");
  }
  if (Source_mask->nline != Source->nline) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->nline != Source->nline");
  }
  if (Source_mask->nsample != Source->nsample) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->nsample != Source->nsample");
  }
  if (Source_mask->datatype != MTKe_uint8) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->datatype != MTKe_uint8");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regressed == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Regressed == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regressed == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regressed_mask == NULL                             */
  /* ------------------------------------------------------------------ */

  if (Regressed_mask == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regressed_mask == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Allocate memory for output data.                                   */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(Source->nline, Source->nsample, 
				 MTKe_float, &regressed_tmp);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(Source->nline, Source->nsample, 
				 MTKe_uint8, &regressed_mask_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Resample regression coefficients to resolution of source data.     */
  /* ------------------------------------------------------------------ */

  status = MtkResampleRegressionCoeff(Regression_coeff,
				      Regression_coeff_map_info, 
				      Source_map_info, 
				      &regression_coeff_resampled);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* For each valid pixel in source map...                              */
  /* ------------------------------------------------------------------ */
 
  for (iline = 0 ; iline < Source->nline ; iline++) {
    for (isample = 0 ; isample < Source->nsample ; isample++) {
      if (Source_mask->data.u8[iline][isample]) {

  /* ------------------------------------------------------------------ */
  /* Appy regression at this pixel.                                     */
  /* ------------------------------------------------------------------ */

	if (regression_coeff_resampled.valid_mask.data.u8[iline][isample]) {
	  regressed_tmp.data.f[iline][isample] = 
	    Source->data.f[iline][isample] *
	    regression_coeff_resampled.slope.data.f[iline][isample] + 
	    regression_coeff_resampled.intercept.data.f[iline][isample];
	  regressed_mask_tmp.data.u8[iline][isample] = 1;
	}

  /* ------------------------------------------------------------------ */
  /* End loop for each valid pixel in source map.                       */
  /* ------------------------------------------------------------------ */

      }
    }
  }
 
  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

  status = MtkRegressionCoeffFree(&regression_coeff_resampled);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Regressed = regressed_tmp;
  *Regressed_mask = regressed_mask_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&regressed_tmp);
  MtkDataBufferFree(&regressed_mask_tmp);
  return status_code;
}


