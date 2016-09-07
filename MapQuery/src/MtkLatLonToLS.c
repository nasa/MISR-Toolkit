/*===========================================================================
=                                                                           =
=                              MtkLatLonToLS                                =
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

/** \brief Convert decimal degrees latitude and longitude to line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert decimal degrees latitude 44.322089 and longitude -112.00821 to line, sample.
 *
 *  \code
 *  status = MtkLatLonToLS(mapinfo, 44.322089, -112.00821, &line, &sample);
 *  \endcode
 */

MTKt_status MtkLatLonToLS(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  double lat_dd,        /**< [IN] Latitude */
  double lon_dd,        /**< [IN] Longitude */
  float *line,          /**< [OUT] Line */
  float *sample         /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  double som_x;                 /* SOM X */
  double som_y;                 /* SOM Y */

  if (line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkLatLonToSomXY(mapinfo.path, lat_dd, lon_dd, &som_x, &som_y);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLS(mapinfo, som_x, som_y, line, sample);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (status_code != MTK_NULLPTR) {
    *line = -1.0;
    *sample = -1.0;
  }
  return status_code;
}
