/*===========================================================================
=                                                                           =
=                       MtkSetRegionByPathBlockRange                        =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrSetRegion.h"
#include "MisrCoordQuery.h"
#include "MisrError.h"
#include "MisrUtil.h"
#include "MisrOrbitPath.h"
#include "MisrProjParam.h"

/** \brief Select region by path and block range
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we select the region on path 39 starting at block 50 and ending at block 60. 
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByPathBlockRange(39, 50, 60, &region);
 *  \endcode
 */

MTKt_status MtkSetRegionByPathBlockRange(
  int path_number,    /**< [IN] Path */
  int start_block,    /**< [IN] Start Block */
  int end_block,      /**< [IN] End Block */
  MTKt_Region *region /**< [OUT] Region */ )
{
  MTKt_status status_code;	/* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_Region rgn;		/* Region structure */
  int block;			/* Block loop index */
  double min_som_x;		/* Minimum SOM X */
  double max_som_x;		/* Maximum SOM X */
  double min_som_y;		/* Minimum SOM Y */
  double max_som_y;		/* Maximum SOM Y */
  double ulc_som_x;	        /* Upper left SOM X */
  double ulc_som_y;             /* Upper left SOM Y */
  double lrc_som_x;             /* Lower right SOM X */
  double lrc_som_y;             /* Lower right SOM Y */

  if (region == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check path bounds */
  if (path_number < 1 || path_number > 233)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check block bounds */
  if (start_block > end_block)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (start_block < 1 || end_block > 180)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Set starting som bounds to the coordinates of the first block  */
  status = MtkBlsToSomXY(path_number, MAXRESOLUTION, start_block,
			 0.0, 0.0, &min_som_x, &min_som_y);
  MTK_ERR_COND_JUMP(status);

  status = MtkBlsToSomXY(path_number, MAXRESOLUTION, start_block,
			 MAXNLINE-1, MAXNSAMPLE-1, &max_som_x, &max_som_y);
  MTK_ERR_COND_JUMP(status);

  /* Expand the coordinates appropriately for each block */
  for (block = start_block+1; block <= end_block; block++) {

    status = MtkBlsToSomXY(path_number, MAXRESOLUTION, block,
			   0.0, 0.0, &ulc_som_x, &ulc_som_y);
    MTK_ERR_COND_JUMP(status);
    status = MtkBlsToSomXY(path_number, MAXRESOLUTION, block,
			   MAXNLINE-1, MAXNSAMPLE-1, &lrc_som_x, &lrc_som_y);
    MTK_ERR_COND_JUMP(status);

#ifdef DEBUG
    printf("              %f                               %f\n", ulc_som_x, min_som_x);
    printf("%f                  %f    %f               %f\n", ulc_som_y, lrc_som_y, min_som_y, max_som_y);
    printf("              %f                               %f\n", lrc_som_x, max_som_x);
#endif

    if (ulc_som_x < min_som_x) min_som_x = ulc_som_x;
    if (lrc_som_x > max_som_x) max_som_x = lrc_som_x;
    if (ulc_som_y < min_som_y) min_som_y = ulc_som_y;
    if (lrc_som_y > max_som_y) max_som_y = lrc_som_y;
  }

#ifdef DEBUG
  {
    double ulclon, ulclat, lrclon, lrclat;

    MtkSomXYToLatLon(path_number,min_som_x,min_som_y, &ulclat,&ulclon);
    MtkSomXYToLatLon(path_number,max_som_x,max_som_y, &lrclat,&lrclon);

    printf("              %f                               %f\n", ulclat, min_som_x);
    printf("%f                  %f    %f               %f\n", ulclon, lrclon, min_som_y, max_som_y);
    printf("              %f                                %f\n", lrclat, max_som_x);
  }
#endif

  /* Determine half of the extent in som x/y for entire block range */
  /* hextent is measured from edge pixel center to edge pixel center */
  rgn.hextent.xlat = (max_som_x - min_som_x) / 2.0;
  rgn.hextent.ylon = (max_som_y - min_som_y) / 2.0;

  if (rgn.hextent.xlat <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  if (rgn.hextent.ylon <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Determine center in geographic coordinates */
  status = MtkSomXYToLatLon(path_number, min_som_x + rgn.hextent.xlat,
			    min_som_y + rgn.hextent.ylon,
			    &(rgn.geo.ctr.lat), &(rgn.geo.ctr.lon));
  MTK_ERR_COND_JUMP(status);

  *region = rgn;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
