/*===========================================================================
=                                                                           =
=                          MtkResampleCubicConvolution                      =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2008, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include "MisrUtil.h"
#include <stdlib.h>
#include <math.h>

float kernel(float d, float a) {
  if (d >= 0.0 && d <= 1.0) {
    return ((a+2.0) * d * d * d - (a+3.0) * d * d + 1.0);
  }
  if (d > 1.0 && d <= 2.0) {
    return (a * d * d * d - 5.0 * a * d * d + 8.0 * a * d - 4.0 * a);
  }
  return 0.0;
}

/** \brief  Resample source data at the given coordinates using interpolation by cubic convolution.
 *
 *  Convolution kernel used in this module is described in [1]
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example: 
 *  In this example we resample source with source_mask and coordinates in line and sample using cubic convolution. Resampled output is in resampled and resampled_mask.
 *
 *  \code
 *  status = MtkResampleCubicConvolution(&source, &source_mask, &line, &sample, a, &resampled, &resampled_mask);
 *  \endcode
 *
 *  \note
 *
 * References:
 *
 * [1] Keys, "Cubic Convolution Interpolation for Digital Image Processing", IEEE Transactions on Acoustics, Speech, and Signal Processing, Vol. ASSP-29, NO. 6, December 1981.
 */
MTKt_status MtkResampleCubicConvolution(
  const MTKt_DataBuffer *Source, /**< [IN] Source data.  (float) */
  const MTKt_DataBuffer *Source_mask, /**< [IN] Valid mask for source data. (uint8)*/
  const MTKt_DataBuffer *Line, /**< [IN] Line coordinates.  (float) */
  const MTKt_DataBuffer *Sample, /**< [IN] Sample coordinates. (float) */
  float A,			 /**< [IN] Convolution parameter (-1.0 <= A <= 0.0) */
  MTKt_DataBuffer *Resampled, /**< [OUT] Resampled data. (float)*/
  MTKt_DataBuffer *Resampled_mask /**< [OUT] Valid mask for resampled data */
) 
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_DataBuffer resampled_tmp = MTKT_DATABUFFER_INIT;
				/* Resampled data */
  MTKt_DataBuffer resampled_mask_tmp = MTKT_DATABUFFER_INIT;
				/* Valid mask for resampled data */
  int iline;
  int isample;

  /* ------------------------------------------------------------------ */
  /* Argument check: Source == NULL                                     */
  /*                 Source->nline < 1                                  */
  /*                 Source->nsample < 1                                */
  /*                 Source->datatype == MTKe_float                     */
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
  /* Argument check: Line == NULL                                       */
  /*                 Line->nline < 1                                    */
  /*                 Line->nsample < 1                                  */
  /*                 Line->datatype != MTKe_float                       */
  /* ------------------------------------------------------------------ */

  if (Line == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Line == NULL");
  }
  if (Line->nline < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Line->nline < 1");
  }
  if (Line->nsample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Line->nsample < 1");
  }
  if (Line->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Line->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Sample == NULL                                     */
  /*                 Sample->nline != Line->nline                       */
  /*                 Sample->nsample != Line->nsample                   */
  /*                 Sample->datatype != MTKe_float                     */
  /* ------------------------------------------------------------------ */

  if (Sample == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Sample == NULL");
  }
  if (Sample->nline != Line->nline) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Sample->nline != Line->nline");
  }
  if (Sample->nsample != Line->nsample) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Sample->nsample != Line->nsample");
  }
  if (Sample->datatype != MTKe_float) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Sample->datatype != MTKe_float");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: A > 0.0                                            */
  /*                 A < -1.0                                           */
  /* ------------------------------------------------------------------ */

  if (A > 0.0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"A > 0.0");
  }
  if (A < -1.0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"A < -1.0");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Resampled == NULL                                  */
  /* ------------------------------------------------------------------ */

  if (Resampled == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Resampled == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Resampled_mask == NULL                             */
  /* ------------------------------------------------------------------ */

  if (Resampled_mask == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Resampled_mask == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Allocate memory for resampled data.                                */
  /* ------------------------------------------------------------------ */

  status = MtkDataBufferAllocate(Line->nline, Line->nsample, 
				 MTKe_float, &resampled_tmp);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(Line->nline, Line->nsample, 
				 MTKe_uint8, &resampled_mask_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* For each location to resample...                                   */
  /* ------------------------------------------------------------------ */

  for (iline = 0; iline < Line->nline ; iline++) {
    for (isample = 0; isample < Line->nsample; isample++) {
      float pline = Line->data.f[iline][isample]; 
				/* Location to resample. */
      float psample = Sample->data.f[iline][isample];
				/* Location to resample. */
      int pline_int = (int)floor(pline + 0.5);
				/* Integer pixel location containing location
				   to resample. */
      int psample_int = (int)floor(psample + 0.5);
				/* Integer pixel location containing location
				   to resample. */
      float missing_default; 	/* Default value to use for missing data. */
      int start_line; 		/* Convolution window start. */
      int start_sample;		/* Convolution window start. */
      int end_line; 		/* Convolution window end. */
      int end_sample;		/* Convolution window end. */
      int wline; 		/* Iterator for convolution window. */
      int wsample; 		/* Iterator for convolution window. */
      float resampled_value = 0.0; 
				/* Resampled value. */

  /* ------------------------------------------------------------------ */
  /* If the source pixel containing this point is not valid, then skip  */
  /* this point.                                                        */
  /* ------------------------------------------------------------------ */

      if (pline_int < 0 || pline_int >= Source->nline ||
	  psample_int < 0 || psample_int >= Source->nsample) {
	continue;
      }

      if (!Source_mask->data.u8[pline_int][psample_int]) {
	continue;
      }

  /* ------------------------------------------------------------------ */
  /* Use the source pixel containing this point at the default value    */
  /* for missing data in the convolution window.                        */
  /* ------------------------------------------------------------------ */

      missing_default = Source->data.f[pline_int][psample_int];

  /* ------------------------------------------------------------------ */
  /* Determine the convolution window for resampling this point.        */
  /* ------------------------------------------------------------------ */

      start_line = (int)floor(pline-1);
      start_sample = (int)floor(psample-1);
      end_line = start_line + 4;
      end_sample = start_sample + 4;

  /* ------------------------------------------------------------------ */
  /* For each point in the convolution window...                        */
  /* ------------------------------------------------------------------ */

      for (wline = start_line ; wline < end_line; wline++) {
	for (wsample = start_sample ; wsample < end_sample; wsample++) {
	  float wvalue; 	/* Source value at this point 
				   in the convolution window. */
	  int valid = 1;	/* Flag indicating if data is valid 
				   at this point in the convolution 
				   window. */
	  float dline;	        /* Distance from this point to interpolation
				   point. */
	  float dsample; 	/* Distance from this point to interpolation 
				   point. */

  /* ------------------------------------------------------------------ */
  /* Get the source data value for this point.  If the point is not     */
  /* valid use the default for missing data (above).                    */
  /* ------------------------------------------------------------------ */

	  if (wline < 0 || wline >= Source->nline ||
	      wsample < 0 || wsample >= Source->nsample) {
	    valid = 0;
	  } else {
	    if (!Source_mask->data.u8[wline][wsample]) {
	      valid = 0;
	    }
	  }

	  if (valid) {
	    wvalue = Source->data.f[wline][wsample];
	  } else {
	    wvalue = missing_default;
	  }

  /* ------------------------------------------------------------------ */
  /* Calculate distance from this point to the interpolation point.     */
  /* ------------------------------------------------------------------ */

	  dline = fabs((float)wline - pline);
	  dsample = fabs((float)wsample - psample);

  /* ------------------------------------------------------------------ */
  /* Calculate contribution to resampled value for this point.          */	 
  /* ------------------------------------------------------------------ */

	  resampled_value += wvalue * kernel(dline,A) * kernel(dsample,A);

  /* ------------------------------------------------------------------ */
  /* End loop for each point in convolution window.                     */
  /* ------------------------------------------------------------------ */

	}
      }

  /* ------------------------------------------------------------------ */
  /* Set resampled value at this point.                                 */
  /* ------------------------------------------------------------------ */

      resampled_tmp.data.f[iline][isample] = resampled_value;
      resampled_mask_tmp.data.u8[iline][isample] = 1;

  /* ------------------------------------------------------------------ */
  /* End loop for each location to resample.                            */
  /* ------------------------------------------------------------------ */

    }
  }

  *Resampled = resampled_tmp;
  *Resampled_mask = resampled_mask_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&resampled_tmp);
  MtkDataBufferFree(&resampled_mask_tmp);
  return status_code;
}


