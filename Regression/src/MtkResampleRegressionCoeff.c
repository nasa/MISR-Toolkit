/*===========================================================================
=                                                                           =
=                          MtkResampleRegressionCoeff                       =
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
#include "MisrReProject.h"
#include "MisrUtil.h"
#include <stdlib.h>
#include <math.h>

/** \brief  Resample regression coefficients at each pixel in the target map.  Resampling is by cubic convolution.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example we use cubic convolution to resample regression_coeff with regression_coeff_map_info to target_map_info. Output is at regression_coeff_out.
 *
 *  \code
 *  status = MtkResampleRegressionCoeff(&regression_coeff, &regression_coeff_map_info, &target_map_info, &regression_coeff_out);
 *  \endcode
 *
 *  \note
 *  MtkRegressionCoeffCalc() can be used to generate regression_coeff and regression_coeff_map_info from 2 data sets and masks.
 */

MTKt_status MtkResampleRegressionCoeff(
  const MTKt_RegressionCoeff *Regression_coeff, /**< [IN] Regression coefficients. */
  const MTKt_MapInfo *Regression_coeff_map_info, /**< [IN] Map info for regression coefficients. */
  const MTKt_MapInfo *Target_map_info, /**< [IN] Map info for target grid. */
  MTKt_RegressionCoeff *Regression_coeff_out /**< [OUT] Regression coefficients resampled to target grid. */
) 
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_RegressionCoeff regression_coeff_out_tmp = MTKT_REGRESSION_COEFF_INIT;
  MTKt_DataBuffer latitude; 	/* Array if latitude values for the target grid. */
  MTKt_DataBuffer longitude; 	/* Array if longitude values for the target grid. */
  MTKt_DataBuffer line = MTKT_DATABUFFER_INIT;
				/* Array of line coordinates in input regression 
				   coefficients grid. */
  MTKt_DataBuffer sample = MTKT_DATABUFFER_INIT;
				/* Array of sample coordinates in input regression 
				   coefficients grid. */
  MTKt_DataBuffer valid_mask_not_used = MTKT_DATABUFFER_INIT;
				/* Unused valid mask. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff == NULL                           */
  /* ------------------------------------------------------------------ */

  if (Regression_coeff == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regression_coeff == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Regression_coeff_out == NULL                       */
  /* ------------------------------------------------------------------ */

  if (Regression_coeff_out == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Regression_coeff_out == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Calculate latitude/longitude for each pixel in the target grid.    */
  /* ------------------------------------------------------------------ */

  status = MtkCreateLatLon(*Target_map_info, &latitude, &longitude);
  MTK_ERR_COND_JUMP(status);
  
  /* ------------------------------------------------------------------ */
  /* Calculate line/sample coordinates in the regression coefficient    */
  /* grid.                                                              */
  /* ------------------------------------------------------------------ */

  status = MtkTransformCoordinates(*Regression_coeff_map_info, 
				   latitude, longitude, &line, &sample);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);

  /* ------------------------------------------------------------------ */
  /* Resample regression coefficients to target grid.                   */
  /* ------------------------------------------------------------------ */
  
  status = MtkResampleCubicConvolution(&(Regression_coeff->slope), 
				       &(Regression_coeff->valid_mask), 
				       &line, 
				       &sample, 
				       -0.5,
				       &(regression_coeff_out_tmp.slope),
				       &(regression_coeff_out_tmp.valid_mask));
  MTK_ERR_COND_JUMP(status);

  status = MtkResampleCubicConvolution(&(Regression_coeff->intercept), 
				       &(Regression_coeff->valid_mask), 
				       &line, 
				       &sample, 
				       -0.5,
				       &(regression_coeff_out_tmp.intercept),
				       &(valid_mask_not_used));
  MTK_ERR_COND_JUMP(status);

  MtkDataBufferFree(&valid_mask_not_used);

  status = MtkResampleCubicConvolution(&(Regression_coeff->correlation),
				       &(Regression_coeff->valid_mask),
				       &line,
				       &sample,
				       -0.5,
				       &(regression_coeff_out_tmp.correlation),
				       &(valid_mask_not_used));
  MTK_ERR_COND_JUMP(status);

  MtkDataBufferFree(&valid_mask_not_used);

  /* ------------------------------------------------------------------ */
  /* Free memory.                                                       */
  /* ------------------------------------------------------------------ */

  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Regression_coeff_out = regression_coeff_out_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&latitude);
  MtkDataBufferFree(&longitude);
  MtkDataBufferFree(&line);
  MtkDataBufferFree(&sample);
  MtkDataBufferFree(&valid_mask_not_used);
  return status_code;
}
