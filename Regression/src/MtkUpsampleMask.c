/*===========================================================================
=                                                                           =
=                            MtkUpsampleMask                                =
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

/** \brief  Upsample a mask by nearest neighbor sampling.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example we upsample source_mask by a factor of size_factor. The output is result_mask.
 *
 *  \code
 *  status = MtkUpsampleMask(&source_mask, size_factor, &result_mask);
 *  \endcode
 *
 *  \note
 */

MTKt_status MtkUpsampleMask(
  const MTKt_DataBuffer *Source_mask, /**< [IN] Source mask. (uint8) */
  int Size_factor, /**< [IN] Number of pixels to expand along each dimension. */
  MTKt_DataBuffer *Result_mask /**< [OUT] Upsampled result. (uint8) */
) 
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status of called routines. */
  MTKt_DataBuffer result_mask_tmp = MTKT_DATABUFFER_INIT;
				/* Valid mask for result. */
  int number_line_out; 		/* Number of lines in output grid.*/
  int number_sample_out; 	/* Number of samples in output grid.*/
  int iline;			/* Loop iterator. */
  int isample;			/* Loop iterator. */

  /* ------------------------------------------------------------------ */
  /* Argument check: Source_mask == NULL                                */
  /*                 Source_mask->nline < 1                             */
  /*                 Source_mask->nsample < 1                           */
  /*                 Source_mask->datatype = MTKe_uint8                 */
  /* ------------------------------------------------------------------ */

  if (Source_mask == NULL) {
    MTK_ERR_CODE_MSG_JUMP(MTK_NULLPTR,"Source_mask == NULL");
  }
  if (Source_mask->nline < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->nline < 1");
  }
  if (Source_mask->nsample < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->nsample < 1");
  }
  if (Source_mask->datatype != MTKe_uint8) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Source_mask->datatype != MTKe_uint8");
  }

  /* ------------------------------------------------------------------ */
  /* Argument check:                                                    */
  /*   Size_factor < 1                                                  */
  /* ------------------------------------------------------------------ */
  
  if (Size_factor < 1) {
    MTK_ERR_CODE_MSG_JUMP(MTK_OUTBOUNDS,"Size_factor < 1");
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

  number_line_out = Source_mask->nline * Size_factor;
  number_sample_out = Source_mask->nsample * Size_factor; 

  /* ------------------------------------------------------------------ */
  /* Allocate memory for result                                         */
  /* ------------------------------------------------------------------ */
  
  status = MtkDataBufferAllocate(number_line_out, number_sample_out,
				 MTKe_uint8, &result_mask_tmp);
  MTK_ERR_COND_JUMP(status);

  /* ------------------------------------------------------------------ */
  /* For each location in result grid...                                */
  /* ------------------------------------------------------------------ */

  for (iline = 0; iline < number_line_out ; iline++) {
    for (isample = 0; isample < number_sample_out; isample++) {


  /* ------------------------------------------------------------------ */
  /* Duplicate the corresponding value in the source grid.              */
  /* ------------------------------------------------------------------ */

      result_mask_tmp.data.u8[iline][isample] = 
	Source_mask->data.u8[iline / Size_factor][isample / Size_factor];

  /* ------------------------------------------------------------------ */
  /* End loop for each location in result grid.                         */
  /* ------------------------------------------------------------------ */
  
    }
  }

  /* ------------------------------------------------------------------ */
  /* Return.                                                            */
  /* ------------------------------------------------------------------ */

  *Result_mask = result_mask_tmp;
  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkDataBufferFree(&result_mask_tmp);
  return status_code;
}
