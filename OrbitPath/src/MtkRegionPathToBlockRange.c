/*===========================================================================
=                                                                           =
=                         MtkRegionPathToBlockRange                         =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrOrbitPath.h"
#include "MisrCoordQuery.h"
#include "MisrProjParam.h"
#include "MisrError.h"
#include <stdlib.h>

#define ULI 0			/* Upper left corner index */
#define LRI 1			/* Lower right corner index */

/** \brief Get start and end block numbers of a path over a particular region
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we get the start and end block numbers of path 78 for a particular region of interest.
 *
 *  \code
 *  status = MtkRegionPathToBlockRange(region, 78, &start_block, &end_block);
 *  \endcode
 */

MTKt_status MtkRegionPathToBlockRange(
  MTKt_Region region,   /**< [IN] Region */
  int path,             /**< [IN] Path */
  int *start_block,     /**< [OUT] Start Block */
  int *end_block        /**< [OUT] End Block */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  int b;			/* Block */
  int n = 0;			/* Block count */
  int sb = 0;			/* Start block */
  MTKt_SomRegion rgnsom;	/* Region Som coordinates */
  MTKt_SomCoord blksom[2];	/* Block Som coordinate */
  MTKt_boolean found = MTK_FALSE; /* Found a corner in the region flag */

  if (start_block == NULL || end_block == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* -------------------------------------------------------------------- */
  /* Determine SOM coordinates of the corners of the region for this path */
  /* -------------------------------------------------------------------- */

  status = MtkLatLonToSomXY(path, region.geo.ctr.lat, region.geo.ctr.lon,
		            &rgnsom.ctr.x, &rgnsom.ctr.y);
  MTK_ERR_COND_JUMP(status);
  rgnsom.path = path;
  rgnsom.ulc.x = rgnsom.ctr.x - region.hextent.xlat;
  rgnsom.ulc.y = rgnsom.ctr.y - region.hextent.ylon;
  rgnsom.lrc.x = rgnsom.ctr.x + region.hextent.xlat;
  rgnsom.lrc.y = rgnsom.ctr.y + region.hextent.ylon;

  /* --------------------------------------------------------------- */
  /* Loop over each block of the to find a corner inside the region  */
  /* --------------------------------------------------------------- */

  for (b = 1; b <= NBLOCK; b++) {

    /* Determine ulc and lrc som coordinates for all this block */
    status = MtkBlsToSomXY(path, MAXRESOLUTION, b, 0.0, 0.0,
			   &blksom[ULI].x, &blksom[ULI].y);
    MTK_ERR_COND_JUMP(status);

    status = MtkBlsToSomXY(path, MAXRESOLUTION, b, MAXNLINE-1, MAXNSAMPLE-1,
			   &blksom[LRI].x, &blksom[LRI].y);
    MTK_ERR_COND_JUMP(status);

    /* Checks if ulc of the block is inside the region */
    if (blksom[ULI].x >= rgnsom.ulc.x && blksom[ULI].x <= rgnsom.lrc.x &&
	blksom[ULI].y >= rgnsom.ulc.y && blksom[ULI].y <= rgnsom.lrc.y)

      found = MTK_TRUE;

    /* Checks if urc of the block is inside the region */
    else if (blksom[ULI].x >= rgnsom.ulc.x && blksom[ULI].x <= rgnsom.lrc.x &&
	     blksom[LRI].y >= rgnsom.ulc.y && blksom[LRI].y <= rgnsom.lrc.y)

      found = MTK_TRUE;

    /* Checks if lrc of the block is inside the region */
    else if (blksom[LRI].x >= rgnsom.ulc.x && blksom[LRI].x <= rgnsom.lrc.x &&
	     blksom[LRI].y >= rgnsom.ulc.y && blksom[LRI].y <= rgnsom.lrc.y)

      found = MTK_TRUE;

    /* Checks if llc of the block is inside the region */
    else if (blksom[LRI].x >= rgnsom.ulc.x && blksom[LRI].x <= rgnsom.lrc.x &&
	     blksom[ULI].y >= rgnsom.ulc.y && blksom[ULI].y <= rgnsom.lrc.y)

      found = MTK_TRUE;

    /* Checks if ulc of the region is inside this block */
    else if (rgnsom.ulc.x >= blksom[ULI].x && rgnsom.ulc.x <= blksom[LRI].x &&
	     rgnsom.ulc.y >= blksom[ULI].y && rgnsom.ulc.y <= blksom[LRI].y)

      found = MTK_TRUE;

    /* Checks if lrc of the region is inside this block */
    else if (rgnsom.lrc.x >= blksom[ULI].x && rgnsom.lrc.x <= blksom[LRI].x &&
	     rgnsom.lrc.y >= blksom[ULI].y && rgnsom.lrc.y <= blksom[LRI].y)

      found = MTK_TRUE;

    /* Checks if block straddles the region horizontally */
    if (((blksom[ULI].x >= rgnsom.ulc.x && blksom[ULI].x <= rgnsom.lrc.x) ||
	 (blksom[LRI].x >= rgnsom.ulc.x && blksom[LRI].x <= rgnsom.lrc.x)) &&
	(blksom[ULI].y < rgnsom.ulc.y && blksom[LRI].y > rgnsom.lrc.y))

      found = MTK_TRUE;

    /* Checks if region straddles the block horizontally */
    if (((rgnsom.ulc.x >= blksom[ULI].x && rgnsom.ulc.x <= blksom[LRI].x) ||
	 (rgnsom.lrc.x >= blksom[ULI].x && rgnsom.lrc.x <= blksom[LRI].x)) &&
	(rgnsom.ulc.y < blksom[ULI].y && rgnsom.lrc.y > blksom[LRI].y))

      found = MTK_TRUE;

    if (found) {
      found = MTK_FALSE;
      if (sb == 0) sb = b;
      n++;
    }
  }

  *start_block = sb;
  *end_block = sb + n - 1;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
