/*===========================================================================
=                                                                           =
=                       MtkSetRegionByPathSomUlcLrc                         =
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

/** \brief Select region by Path and SOM X/Y of upper left corner and
 *         lower right corner in meters.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we select the region with upper left SOM X 15600000.0 meters, SOM Y -300.0 meters and lower right SOM X 16800000.0 meters, SOM Y 2000.0 meters for path 27.
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByPathSomUlcLrc(27, 15600000.0, -300.0, 16800000.0, 2000.0, &region);
 *  \endcode
 */

MTKt_status MtkSetRegionByPathSomUlcLrc(
  int path,         /**< [IN] Path */
  double ulc_som_x, /**< [IN] Upper left corner SOM X */
  double ulc_som_y, /**< [IN] Upper left corner SOM Y */
  double lrc_som_x, /**< [IN] Lower right corner SOM X */
  double lrc_som_y, /**< [IN] Lower right corner SOM Y */
  MTKt_Region *region /**< [OUT] Region */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;      /* Return status of this function */
  MTKt_Region rgn;		/* Region structure */
  double ctr_som_x;		/* Center of region in SOM X */
  double ctr_som_y;		/* Center of region in SOM Y */

  if (region == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check path bounds */
  if (path < 1 || path > 233)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check SOM X bounds */
  if (lrc_som_x < ulc_som_x)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check SOM Y bounds */
  if (lrc_som_y < ulc_som_y)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Determine center of region in lat/lon coordinates */

  ctr_som_x = ulc_som_x + ((lrc_som_x - ulc_som_x) / 2.0);
  ctr_som_y = ulc_som_y + ((lrc_som_y - ulc_som_y) / 2.0);

  status = MtkSomXYToLatLon(path, ctr_som_x, ctr_som_y,
			    &rgn.geo.ctr.lat, &rgn.geo.ctr.lon);
  MTK_ERR_COND_JUMP(status);

  /* Half of the over all extent to measure from center */
  rgn.hextent.xlat = (lrc_som_x - ulc_som_x + MAXRESOLUTION) / 2.0;
  rgn.hextent.ylon = (lrc_som_y - ulc_som_y + MAXRESOLUTION) / 2.0;

  *region = rgn;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
