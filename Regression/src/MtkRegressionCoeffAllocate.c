/*===========================================================================
=                                                                           =
=                          MtkRegressionCoeffAllocate                       =
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

/** \brief Allocate buffer to contain regression coefficients
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we allocate a buffer with a size of 5 lines by 10 samples of type \c MTKe_int16
 *
 *  \code
 *  status = MtkRegressionCoeffAllocate(5, 10, &regressbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkRegressionCoeffFree() to free the memory used by \c regressbuf
 */

MTKt_status MtkRegressionCoeffAllocate(
  int nline,   /**< [IN] Number of lines */
  int nsample, /**< [IN] Number of samples */
  MTKt_RegressionCoeff *regressbuf /**< [OUT] Data Buffer */ 
)
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status of called routines */
  MTKt_RegressionCoeff rbuf = MTKT_REGRESSION_COEFF_INIT;
				/* Regression buffer structure */
  
  if (regressbuf == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nline < 1)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (nsample < 1)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  status = MtkDataBufferAllocate(nline, nsample, MTKe_uint8, &rbuf.valid_mask);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(nline, nsample, MTKe_float, &rbuf.slope);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(nline, nsample, MTKe_float, &rbuf.intercept);
  MTK_ERR_COND_JUMP(status);

  status = MtkDataBufferAllocate(nline, nsample, MTKe_float, &rbuf.correlation);
  MTK_ERR_COND_JUMP(status);

  *regressbuf = rbuf;

  return MTK_SUCCESS;

ERROR_HANDLE:
  MtkRegressionCoeffFree(&rbuf);
  return status_code;
}
