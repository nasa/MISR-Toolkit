/*===========================================================================
=                                                                           =
=                     MtkPathBlockRangeToBlockCorners                       =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrCoordQuery.h"
#include "MisrError.h"
#include "MisrProjParam.h"

/** \brief Compute block corner coordinates in decimal degrees of latitude and 
 *  longitude for a given path and block range. Coordinates returned are with
 *  respect to the pixel center of the upper left corner, center and lower right 
 *  corner of each block.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we want the block corners for path 37 and blocks 25 to 50.
 *
 *  \code
 *  status = MtkPathBlockRangeToBlockCorners(37, 25, 50, &block_corners);
 *  \endcode
 */

MTKt_status MtkPathBlockRangeToBlockCorners(
  int path,                         /**< [IN] Path */
  int start_block,                  /**< [IN] Start Block */
  int end_block,                    /**< [IN] End Block */
  MTKt_BlockCorners *block_corners  /**< [OUT] Longitude Decimal Degrees */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  int i;			/* Block index */
  MTKt_BlockCorners blkcrnrs = MTKT_BLOCKCORNERS_INIT;
				/* Block corners */

  if (block_corners == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check path bounds */
  if (path < 1 || path > 233)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check block bounds */
  if (start_block > end_block)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (start_block < 1 || end_block > 180)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  blkcrnrs.path = path;
  blkcrnrs.start_block = start_block;
  blkcrnrs.end_block = end_block;

  for (i = start_block; i <= end_block; i++) {
    blkcrnrs.block[i].block_number = i;

    status = MtkBlsToLatLon(path, MAXRESOLUTION, i,
			    0.0, 0.0,
			    &(blkcrnrs.block[i].ulc.lat),
			    &(blkcrnrs.block[i].ulc.lon));
    MTK_ERR_COND_JUMP(status);

    status = MtkBlsToLatLon(path, MAXRESOLUTION, i,
			    0.0, MAXNSAMPLE-1,
			    &(blkcrnrs.block[i].urc.lat),
			    &(blkcrnrs.block[i].urc.lon));
    MTK_ERR_COND_JUMP(status);

    status = MtkBlsToLatLon(path, MAXRESOLUTION, i,
			    (MAXNLINE-1) / 2.0, (MAXNSAMPLE-1) / 2.0,
			    &(blkcrnrs.block[i].ctr.lat),
			    &(blkcrnrs.block[i].ctr.lon));
    MTK_ERR_COND_JUMP(status);

    status = MtkBlsToLatLon(path, MAXRESOLUTION, i,
			    MAXNLINE-1, MAXNSAMPLE-1,
			    &(blkcrnrs.block[i].lrc.lat),
			    &(blkcrnrs.block[i].lrc.lon));
    MTK_ERR_COND_JUMP(status);

    status = MtkBlsToLatLon(path, MAXRESOLUTION, i,
			    MAXNLINE-1, 0.0,
			    &(blkcrnrs.block[i].llc.lat),
			    &(blkcrnrs.block[i].llc.lon));
    MTK_ERR_COND_JUMP(status);
  }

  *block_corners = blkcrnrs;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
