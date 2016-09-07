/*===========================================================================
=                                                                           =
=                        MtkResampleNearestNeighbor                         =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrReProject.h"
#include "MisrError.h"
#include "MisrUtil.h"
#include <string.h>
#include <math.h>

/** \brief Perform nearest neighbor resampling.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we resample the srcbuf using nearest neighbor resampling based on linebuf and samplebuf. The result is resampbuf.
 *
 *  \code
 *  status = MtkResampleNearestNeighbor(srcbuf, linebuf, samplebuf, &resampbuf);
 *  \endcode
 *
 *  \note
 *  The caller is responsible for using MtkDataBufferFree() to free the memory used by resampbuf.
*/

MTKt_status MtkResampleNearestNeighbor(
  MTKt_DataBuffer srcbuf,     /**< [IN] Source data buffer */
  MTKt_DataBuffer linebuf,    /**< [IN] Line data buffer */
  MTKt_DataBuffer samplebuf,  /**< [IN] Sample data buffer */
  MTKt_DataBuffer *resampbuf  /**< [OUT] Resampled data buffer */ )
{
  MTKt_status status;           /* Return status */
  MTKt_status status_code;      /* Return code of this function */
  MTKt_DataBuffer destbuf = MTKT_DATABUFFER_INIT;
                                /* Resampled data buffer structure */
  int lsrc;			/* Source line index */
  int ssrc;			/* Source sample index */
  int srcoff;		        /* Data pointer offset into src buffer */
  int srcnum;			/* Number of elements in src buffer */
  int ldest;			/* Destination line index */
  int sdest;			/* Destination sample index */
  int destoff;		        /* Data pointer offset into dest buffer */
  int destnum;		        /* Number of elements in dest buffer */

  /* Check line and sample buffer sizes */
  if (linebuf.nline != samplebuf.nline)
    MTK_ERR_COND_JUMP(MTK_DIMENSION_MISMATCH);
  if (linebuf.nsample != samplebuf.nsample)
    MTK_ERR_COND_JUMP(MTK_DIMENSION_MISMATCH);

  /* Check line and sample buffer datatypes */
  if (linebuf.datatype != MTKe_float)
    MTK_ERR_COND_JUMP(MTK_DATATYPE_MISMATCH);
  if (samplebuf.datatype != MTKe_float)
    MTK_ERR_COND_JUMP(MTK_DATATYPE_MISMATCH);

  /* --------------------------------------------------- */
  /* Allocate buffers of the latitude and longitude data */
  /* --------------------------------------------------- */

  status = MtkDataBufferAllocate(linebuf.nline, linebuf.nsample,
				 srcbuf.datatype, &destbuf);
  MTK_ERR_COND_JUMP(status);

  /* -------------------------------------------------- */
  /* Resample source data buffer using nearest neighbor */
  /* -------------------------------------------------- */

  srcnum = srcbuf.nline * srcbuf.nsample;
  destnum = destbuf.nline * destbuf.nsample;

  for (ldest=0; ldest < destbuf.nline; ldest++) {
    for (sdest=0; sdest < destbuf.nsample; sdest++) {
      lsrc = (int)floorf(linebuf.data.f[ldest][sdest] + 0.5);
      ssrc = (int)floorf(samplebuf.data.f[ldest][sdest] + 0.5);
      srcoff = lsrc * srcbuf.nsample + ssrc;
      destoff = ldest * destbuf.nsample + sdest;
      if (destoff >= 0 && destoff < destnum && srcoff >= 0 && srcoff < srcnum) {
	memcpy((void *)((char *)destbuf.dataptr + (destoff * destbuf.datasize)),
	       (void *)((char *)srcbuf.dataptr + (srcoff * srcbuf.datasize)),
	       destbuf.datasize);
      }
    }
  }

  *resampbuf = destbuf;

  return MTK_SUCCESS;
ERROR_HANDLE:
  MtkDataBufferFree(&destbuf);
  return status_code;
}
