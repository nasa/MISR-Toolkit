/*===========================================================================
=                                                                           =
=                              MtkLSToLatLon                                =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrMapQuery.h"
#include "MisrCoordQuery.h"
#include "MisrError.h"

/** \brief Convert line, sample to decimal degrees latitude and longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert line 1.0 and sample 1.0 to latitude and longitude in decimal degrees.
 *
 *  \code
 *  status = MtkLSToLatLon(mapinfo, 1.0, 1.0, &lat_dd, &lon_dd);
 *  \endcode
 */

MTKt_status MtkLSToLatLon(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  float line,           /**< [IN] Line */
  float sample,         /**< [IN] Sample */
  double *lat_dd,       /**< [OUT] Latitude */
  double *lon_dd        /**< [OUT] Longitude */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  double som_x;                 /* SOM X */
  double som_y;                 /* SOM Y */

  if (lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkLSToSomXY(mapinfo, line, sample, &som_x, &som_y);
  MTK_ERR_COND_JUMP(status);

  status =  MtkSomXYToLatLon(mapinfo.path, som_x, som_y, lat_dd, lon_dd);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
