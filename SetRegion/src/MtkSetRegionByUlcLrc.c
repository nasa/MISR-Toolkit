/*===========================================================================
=                                                                           =
=                           MtkSetRegionByUlcLrc                            =
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

/** \brief Select region by latitude and longitude of upper left corner and
 *         lower right corner in decimal degrees
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we select the region with upper left latitude 40.0, longitude -120.0 and lower right latitude 30.0, longitude -110.0.
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByUlcLrc(40.0, -120.0, 30.0, -110.0, &region);
 *  \endcode
 */

MTKt_status MtkSetRegionByUlcLrc(
  double ulc_lat_dd, /**< [IN] Upper left corner latitude */
  double ulc_lon_dd, /**< [IN] Upper left corner longitude */
  double lrc_lat_dd, /**< [IN] Lower right corner latitude */
  double lrc_lon_dd, /**< [IN] Lower right corner longitude */
  MTKt_Region *region /**< [OUT] Region */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_Region rgn;		/* Region structure */
  double lat_extent_dd;		/* Latitude extent */
  double lon_extent_dd;		/* Longitude extent */
  double meters_per_deg;        /* Meters per degree at equator */

  if (region == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check latitude bounds */
  if (ulc_lat_dd > 90.0 || ulc_lat_dd < -90.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lrc_lat_dd > 90.0 || lrc_lat_dd < -90.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (ulc_lat_dd < lrc_lat_dd)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check longitude bounds */
  if (ulc_lon_dd > 180.0 || ulc_lon_dd < -180.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lrc_lon_dd > 180.0 || lrc_lon_dd < -180.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check for date line crossing */
  lat_extent_dd = ulc_lat_dd - lrc_lat_dd;
  rgn.geo.ctr.lat = ulc_lat_dd - lat_extent_dd / 2.0;
  if (ulc_lon_dd > lrc_lon_dd) {
    lon_extent_dd = 360 + lrc_lon_dd - ulc_lon_dd;	/* Date line crossed */
  } else {
    lon_extent_dd = lrc_lon_dd - ulc_lon_dd;
  }
  rgn.geo.ctr.lon = ulc_lon_dd + lon_extent_dd / 2.0;

  if (lat_extent_dd <= 0.0) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lon_extent_dd <= 0.0) MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Half of the over all extent to measure from center */
  meters_per_deg = 111319.54315;
  rgn.hextent.xlat = lat_extent_dd * meters_per_deg / 2.0;
  rgn.hextent.ylon = lon_extent_dd * meters_per_deg / 2.0;

  *region = rgn;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
