/*===========================================================================
=                                                                           =
=                          MtkSmoothData                                    =
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

/** \brief Smooth the given array with a boxcar average of the specified width.  The algorithm is similar to the IDL smooth routine.  Smoothed data is type float.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example we smooth data with mask valid_mask and smoothing window widths width_line and width_sample. Output is data_smoothed.
 *
 *  \code
 *  status = MtkSmoothData(&data, &valid_mask, width_line, width_sample, &data_smoothed);
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkSmoothData(
  const MTKt_DataBuffer *Data, /**< [IN] Data.  (float) */
  const MTKt_DataBuffer *Valid_mask, /**< [IN] Mask inidicating where data is valid.  (uint8)*/
  int Width_line, /**< [IN] Width of smoothing window along line dimension. Must be and odd number. */
  int Width_sample, /**< [IN] Width of smoothing window along sample dimension. Must be an odd number. */
  MTKt_DataBuffer *Data_smoothed /**< [OUT] Smoothed data. (float) */
)
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_DataBuffer data_smoothed_tmp = MTKT_DATABUFFER_INIT;
  int iline;
  int isample;

  /* -------------------------------------------------------------- */
  /* Argument check: Data == NULL                                   */
  /*                 Data->nline < 1                                */
  /*                 Data->nsample < 1                              */
  /*                 Data->datatype != MTKe_float                  */
  /* -------------------------------------------------------------- */

  if (Data == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }
  if (Data->nline < 1) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Data->nsample < 1) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Data->datatype != MTKe_float) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Valid_mask == NULL                             */
  /*                 Valid_mask->nline != Data->nline               */
  /*                 Valid_mask->nsample != Data->nsample           */
  /*                 Valid_mask->datatype != MTKe_uint8             */
  /* -------------------------------------------------------------- */

  if (Valid_mask == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }
  if (Valid_mask->nline != Data->nline) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Valid_mask->nsample != Data->nsample) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Valid_mask->datatype != MTKe_uint8) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Width_line < 1                                 */
  /*                 Width_line > Data->nline                       */
  /*                 Width_line is not odd                          */
  /* -------------------------------------------------------------- */

  if (Width_line < 1) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Width_line > Data->nline) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if ((Width_line & 1) == 0) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Width_sample < 1                               */
  /*                 Width_sample > Data->nsample                   */
  /*                 Width_sample is not odd                        */
  /* -------------------------------------------------------------- */

  if (Width_sample < 1) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if (Width_sample > Data->nsample) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }
  if ((Width_sample & 1) == 0) {
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  }

  /* -------------------------------------------------------------- */
  /* Argument check: Data_smoothed = NULL                           */
  /* -------------------------------------------------------------- */

  if (Data_smoothed == NULL) {
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);
  }

  /* -------------------------------------------------------------- */
  /* Allocate memory for smoothed data                              */
  /* -------------------------------------------------------------- */

  status = MtkDataBufferAllocate(Data->nline,
				 Data->nsample,
				 MTKe_float,
				 &data_smoothed_tmp);
  MTK_ERR_COND_JUMP(status);

  /* -------------------------------------------------------------- */
  /* For each location in smoothed data....(begin loop)             */
  /* -------------------------------------------------------------- */

  for (iline = 0 ; iline < Data->nline ; iline++) {
    for (isample = 0 ; isample < Data->nsample ; isample++) {
      double window_sum = 0.0;	/* Sum of valid elements in the window */
      int count = 0; 		/* Count of valid elements in the window */
      int wline;
      int wsample;

  /* -------------------------------------------------------------- */
  /* If this location is not valid in the input data, then skip to  */
  /* the next location.   Don't attempt to fill in gaps.            */
  /* -------------------------------------------------------------- */

      if (!Valid_mask->data.u8[iline][isample]) {
	continue;
      }

  /* -------------------------------------------------------------- */
  /* Sum all valid data points in the smoothing window centered at  */
  /* this location.  Where data is not available, duplicate the     */
  /* data value at the center of the window.  This has the effect   */
  /* of giving higher weight to the center of the window where      */
  /* information is not available across the entire window.         */
  /* -------------------------------------------------------------- */

      for (wline = iline - ((Width_line - 1) / 2) ; 
	   wline <= iline + ((Width_line - 1) / 2) ; wline++) {
	for (wsample = isample - ((Width_sample - 1) / 2) ; 
	     wsample <= isample + ((Width_sample - 1) / 2) ; wsample++) {
	  
	  if ((wline >= 0 && wline < Data->nline) &&
	      (wsample >= 0 && wsample < Data->nsample) &&
	      Valid_mask->data.u8[wline][wsample]) {
	    window_sum += Data->data.f[wline][wsample];
	  } else {
	    window_sum += Data->data.f[iline][isample];
	  }
	  count++;
	}
      }
      
  /* -------------------------------------------------------------- */
  /* Set smoothed value at this location.                           */
  /* -------------------------------------------------------------- */

      data_smoothed_tmp.data.f[iline][isample] = 
	window_sum / (double)count;

  /* -------------------------------------------------------------- */
  /* End loop for each location in smoothed data.                   */
  /* -------------------------------------------------------------- */

    }
  }

  *Data_smoothed = data_smoothed_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&data_smoothed_tmp);
  return status_code;
}
