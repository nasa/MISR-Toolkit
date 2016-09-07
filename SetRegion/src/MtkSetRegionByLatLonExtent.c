/*===========================================================================
=                                                                           =
=                        MtkSetRegionByLatLonExtent                         =
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
#include "MisrError.h"
#include "MisrUtil.h"
#include <string.h>
#ifndef _MSC_VER
#include <strings.h>
#endif
#include <stdlib.h>

/** \brief Select region by latitude, longitude in decimal degrees, and extent in specified units of degrees, meters, kilometers, or pixels.
 *
 *  The parameter extent_units is a case insensitive string that can be set to one of the following values:
 *  -# "degrees", "deg", "dd" for degrees;
 *  -# "meters", "m" for meters;
 *  -# "kilometers", "km" for kilometers; and
 *  -# "275m", "275 meters", "1.1km", "1.1 kilometers" for pixels of a specified resolution per pixel.
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example 1:
 *  In this example, we select the region centered at latitude 35.0 and longitude -115.0.  With a latitude extent of 1.5 degrees and longitude extent of 2 degrees.
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByLatLonExtent(35.0, -115.0, 1.5, 2.0, "deg", &region);
 *  \endcode
 *
 *  \par Example 2:
 *  In this example, we select the region centered at latitude 35.0 and longitude -115.0.  With a latitude extent of 5000 meters and longitude extent of 8000 meters.
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByLatLonExtent(35.0, -115.0, 5000.0, 8000.0, "m", &region);
 *  \endcode
 *
 *  \par Example 3:
 *  In this example, we select the region centered at latitude 35.0 and longitude -115.0.  With a latitude extent of 2.2km and longitude extent of 1.1km.
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByLatLonExtent(35.0, -115.0, 2.2, 1.1, "km", &region);
 *  \endcode
 *
 *  \par Example 4:
 *  In this example, we select the region centered at latitude 35.0 and longitude -115.0.  With a latitude extent of 45 (275 meter per pixels) and longitude extent of 100 (275 meter per pixels).
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByLatLonExtent(35.0, -115.0, 45.0, 100.0, "275m", &region);
 *  \endcode
 *
 *  \par Example 5:
 *  In this example, we select the region centered at latitude 35.0 and longitude -115.0.  With a latitude extent of 35 (1.1 km per pixels) and longitude extent of 25 (1.1 km per pixels).
 *
 *  \code
 *  MTKt_region region = MTKT_REGION_INIT;
 *  status = MtkSetRegionByLatLonExtent(35.0, -115.0, 35.0, 25.0, "1.1km", &region);
 *  \endcode
*/

MTKt_status MtkSetRegionByLatLonExtent(
  double ctr_lat_dd, /**< [IN] Latitude */
  double ctr_lon_dd, /**< [IN] Longitude */
  double lat_extent, /**< [IN] Latitude Extent */
  double lon_extent, /**< [IN] Longitude Extent */
  const char *extent_units,/**< [IN] Extent Units (ex. degrees, deg, dd, meters, m, kilometer, km, 275m, 1.1km) */
  MTKt_Region *region /**< [OUT] Region */ )
{
  MTKt_status status_code;      /* Return code of this function */
  MTKt_Region rgn;		/* Region structure */
  double resolution;		/* Resolution */
  char *endptr;			/* End ptr for strod */

  if (region == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check extent_units */
  if (extent_units == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check latitude bounds */
  if (ctr_lat_dd > 90.0 || ctr_lat_dd < -90.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lat_extent <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Check longitude bounds */
  if (ctr_lon_dd > 180.0 || ctr_lon_dd < -180.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (lon_extent <= 0.0)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Set center of region */
  rgn.geo.ctr.lat = ctr_lat_dd;
  rgn.geo.ctr.lon = ctr_lon_dd;

  /* Set half of the overall extent of region in specified units measured from center */
  if (strncasecmp(extent_units, "degrees", strlen(extent_units)) == 0 ||
      strncasecmp(extent_units, "deg", strlen(extent_units)) == 0 ||
      strncasecmp(extent_units, "dd", strlen(extent_units)) == 0) {

    /* Average earth circumference (40075004m + 39940638m) / 2.0 = 40007821m */
    double meters_per_deg = 40007821.0 / 360.0;
    rgn.hextent.xlat = (lat_extent * meters_per_deg) / 2.0;
    rgn.hextent.ylon = (lon_extent * meters_per_deg) / 2.0;

  } else if (strncasecmp(extent_units, "meters", strlen(extent_units)) == 0 ||
	     strncasecmp(extent_units, "m", strlen(extent_units)) == 0) {

    rgn.hextent.xlat = lat_extent / 2.0;
    rgn.hextent.ylon = lon_extent / 2.0;

  } else if (strncasecmp(extent_units, "kilometers", strlen(extent_units)) == 0 ||
	      strncasecmp(extent_units, "km", strlen(extent_units)) == 0) {

    rgn.hextent.xlat = (lat_extent * 1000) / 2.0;
    rgn.hextent.ylon = (lon_extent * 1000) / 2.0;

  } else if ((resolution = strtod(extent_units, &endptr)) != 0) {

    if (strncasecmp(endptr, "meters", strlen(endptr)) == 0  ||
	strncasecmp(endptr, "m", strlen(endptr)) == 0  ||
	strncasecmp(endptr, " meters", strlen(endptr)) == 0  ||
	strncasecmp(endptr, " m", strlen(endptr)) == 0) {

      rgn.hextent.xlat = (lat_extent * resolution) / 2.0;
      rgn.hextent.ylon = (lon_extent * resolution) / 2.0;

    } else if (strncasecmp(endptr, "kilometers", strlen(endptr)) == 0  ||
	strncasecmp(endptr, "km", strlen(endptr)) == 0  ||
	strncasecmp(endptr, " kilometers", strlen(endptr)) == 0  ||
	strncasecmp(endptr, " km", strlen(endptr)) == 0) {

      rgn.hextent.xlat = (lat_extent * resolution * 1000) / 2.0;
      rgn.hextent.ylon = (lon_extent * resolution * 1000) / 2.0;

    } else {

      MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

   }

  } else {

    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  }

  *region = rgn;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
