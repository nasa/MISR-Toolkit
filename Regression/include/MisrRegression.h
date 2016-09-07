/*===========================================================================
=                                                                           =
=                              MisrRegression                               =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#ifndef MISRREGRESSION_H
#define MISRREGRESSION_H

#include "MisrUtil.h"
#include "MisrError.h"
#include "MisrProjParam.h"
#include "MisrMapQuery.h"

typedef struct {
       MTKt_DataBuffer valid_mask;
       MTKt_DataBuffer slope;
       MTKt_DataBuffer intercept;
       MTKt_DataBuffer correlation;
} MTKt_RegressionCoeff;

#define MTKT_REGRESSION_COEFF_INIT { \
    MTKT_DATABUFFER_INIT, \
    MTKT_DATABUFFER_INIT, \
    MTKT_DATABUFFER_INIT, \
}

MTKt_status MtkRegressionCoeffAllocate(
  int nline,   /**< [IN] Number of lines */
  int nsample, /**< [IN] Number of samples */
  MTKt_RegressionCoeff *regressbuf /**< [OUT] Data Buffer */ 
);

MTKt_status MtkRegressionCoeffFree(
  MTKt_RegressionCoeff *regressbuf /**< [IN,OUT] Data Buffer */ 
);

MTKt_status MtkLinearRegressionCalc(
  int Size,   /**< [IN] Size of X and Y arrays */
  const double *X, /**< [IN] X array */
  const double *Y, /**< [IN] Y array */
  const double *YSigma, /** [IN] Uncertainty in Y */
  double *A, /**< [OUT] A */
  double *B, /**< [OUT] B */
  double *Correlation /**< [OUT] Correlation */
);

MTKt_status MtkSmoothData(
  const MTKt_DataBuffer *Data, /**< [IN] Data */
  const MTKt_DataBuffer *Valid_mask, /**< [IN] Mask inidicating where data is valid. */
  int Width_line, /**< [IN] Width of smoothing window along line dimension. Must be and odd number. */
  int Width_sample, /**< [IN] Width of smoothing window along sample dimension. Must be an odd number. */
  MTKt_DataBuffer *Data_smoothed /**< [OUT] Smoothed data */
);

MTKt_status MtkRegressionCoeffCalc(
  const MTKt_DataBuffer *Data1, /**< [IN] Data buffer 1 */
  const MTKt_DataBuffer *Valid_mask1, /**< [IN] Valid mask for data buffer 1 */
  const MTKt_DataBuffer *Data2, /**< [IN] Data buffer 2 */
  const MTKt_DataBuffer *Data2_sigma, /**< [IN] Uncertainty for data buffer 2 */
  const MTKt_DataBuffer *Valid_mask2, /**< [IN] Valid mask for data buffer 2 */
  const MTKt_MapInfo *Map_info, /**< [IN] Map info for input data. */
  int Regression_size_factor, /**< [IN] Number of pixels to aggregate along each axis when generating regression coefficients. */
  MTKt_RegressionCoeff *Regression_coeff, /**< [OUT] Regression coefficients. */
  MTKt_MapInfo *Regression_map_info /**< [OUT] Map info for regression coefficients */
) ;

MTKt_status MtkResampleRegressionCoeff(
  const MTKt_RegressionCoeff *Regression_coeff, /**< [IN] Regression coefficients. */
  const MTKt_MapInfo *Regression_coeff_map_info, /**< [IN] Map info for regression coefficients. */
  const MTKt_MapInfo *Target_map_info, /**< [IN] Map info for target grid. */
  MTKt_RegressionCoeff *Regression_coeff_out /**< [OUT] Regression coefficients resampled to target grid. */
) ;

MTKt_status MtkApplyRegression(
  const MTKt_DataBuffer *Source, /**< [IN] Input data. */
  const MTKt_DataBuffer *Source_mask, /**< [IN] Valid mask for input data */
  const MTKt_MapInfo    *Source_map_info, /**< [IN] Map info for input data */
  const MTKt_RegressionCoeff *Regression_coeff, /**< [IN] Regression coefficients */
  const MTKt_MapInfo *Regression_coeff_map_info, /**< [IN] Map info for regression coefficients */
  MTKt_DataBuffer *Regressed,  /**< [OUT] Output data. */
  MTKt_DataBuffer *Regressed_mask /**< [OUT] Valid mask for output data */
);

MTKt_status MtkDownsample(
  const MTKt_DataBuffer *Source, /**< [IN] Source data. (float) */
  const MTKt_DataBuffer *Source_mask, /**< [IN] Valid mask for source data. (uint8) */
  int Size_factor, /**< [IN] Number of pixels to aggregate along each dimension. */
  MTKt_DataBuffer *Result, /**< [OUT] Downsampled result. (float) */
  MTKt_DataBuffer *Result_mask /**< [OUT] Valid mask for result. (uint8) */
			  );

MTKt_status MtkUpsampleMask(
  const MTKt_DataBuffer *Source_mask, /**< [IN] Source mask. (uint8) */
  int Size_factor, /**< [IN] Number of pixels to expand along each dimension. */
  MTKt_DataBuffer *Result_mask /**< [OUT] Upsampled result. (uint8) */
			);

#endif /* MISRREGRESSION_H */
