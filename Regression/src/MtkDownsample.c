/*===========================================================================
=                                                                           =
=                            MtkDownsample                                  =
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

/** \brief  Downsample data by averaging pixels.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example we downsample source with source_mask by size_factor. The result is stored in result with result_mask.
 *
 *  \code
 *  status = MtkDownsample(&source, &source_mask, size_factor, &result, &result_mask);
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkDownsample(
  const MTKt_DataBuffer *Source, /**< [IN] Source data. (float) */
  const MTKt_DataBuffer *Source_mask, /**< [IN] Valid mask for source data. (uint8) */
  int Size_factor, /**< [IN] Number of pixels to aggregate along each dimension. */
  MTKt_DataBuffer *Result, /**< [OUT] Downsampled result. (float) */
  MTKt_DataBuffer *Result_mask /**< [OUT] Valid mask for result. (uint8) */
) 
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_DataBuffer result_tmp = MTKT_DATABUFFER_INIT; 
				/* Downsampled result. */
  MTKt_DataBuffer result_mask_tmp = MTKT_DATABUFFER_INIT;
				/* Valid mask for result. */
  int number_line_out; 		/* Number of lines in output grid.*/
  int number_sample_out; 	/* Number of samples in output grid.*/
  int iline;			/* Loop iterator. */
  int isample;			/* Loop iterator. */

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
  /* Argument check:                                                    */
  /*   Size_factor < 1                                                  */
  /*   Source->nline % Size_factor != 0                                 */
  /*   Source->nsample % Size_factor != 0                               */
  /* ------------------------------------------------------------------ */
  
  if (Size_factor < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Size_factor < 1");
  }
  if (Source->nline % Size_factor != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source->nline % Size_factor != 0");
  }
  if (Source->nsample % Size_factor != 0) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source->nsample % Size_factor != 0");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Result == NULL                                     */
  /* ------------------------------------------------------------------ */

  if (Result == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Result == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check: Result_mask == NULL                                */
  /* ------------------------------------------------------------------ */

  if (Result_mask == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Result_mask == NULL");
  }

  /* ------------------------------------------------------------------ */
  /* Determine size of output array.                                    */
  /* ------------------------------------------------------------------ */

  number_line_out = Source->nline / Size_factor;
  number_sample_out = Source->nsample / Size_factor; 

  /* ------------------------------------------------------------------ */
  /* Allocate memory for result                                         */
  /* ------------------------------------------------------------------ */
  
  status = MtkDataBufferAllocate(number_line_out, number_sample_out,
				 MTKe_float, &result_tmp);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(number_line_out, number_sample_out,
				 MTKe_uint8, &result_mask_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* For each location in result grid...                                */
  /* ------------------------------------------------------------------ */

  for (iline = 0; iline < number_line_out ; iline++) {
    for (isample = 0; isample < number_sample_out; isample++) {
      int rline_start, rsample_start;
      int rline_end, rsample_end;
      int rline, rsample;
      int count;
      double sum = 0;
      rline_start = iline * Size_factor;
      rsample_start = isample * Size_factor;
      rline_end = rline_start + Size_factor;
      rsample_end = rsample_start + Size_factor;
      count = 0;

  /* ------------------------------------------------------------------ */
  /* Calculate average of source pixels at this location.               */
  /* ------------------------------------------------------------------ */

      for (rline = rline_start ; rline < rline_end ; rline++) {
	for (rsample = rsample_start ; rsample < rsample_end ; rsample++) {
	  if (Source_mask->data.u8[rline][rsample]) {
	    sum += Source->data.f[rline][rsample];
	    count++;
	  }
	}
      }

      if (count > 0) {
	result_tmp.data.f[iline][isample] = sum / count;
	result_mask_tmp.data.u8[iline][isample] = 1;
      } else {
	result_mask_tmp.data.u8[iline][isample] = 0;
      }

  /* ------------------------------------------------------------------ */
  /* End loop for each location in result grid.                         */
  /* ------------------------------------------------------------------ */
  
    }
  }

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Result = result_tmp;
  *Result_mask = result_mask_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&result_tmp);
  MtkDataBufferFree(&result_mask_tmp);
  return status_code;
}
