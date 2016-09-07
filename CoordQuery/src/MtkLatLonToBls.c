/*===========================================================================
=                                                                           =
=                               MtLatLonToBls                               =
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
#include "proj.h"

/** \brief Convert decimal degrees latitude and longitude to block, line,
 *         sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert latitude 82.740690 and longitude -3.310459 for path 1 at 1100 meter resolution to block, line, sample.
 *
 *  \code
 *  status = MtkLatLonToBls(1, 1100, 82.740690, -3.310459, &block, &line, &sample);
 *  \endcode
 */

MTKt_status MtkLatLonToBls(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  double lat_dd,         /**< [IN] Latitude */
  double lon_dd,         /**< [IN] Longitude */
  int *block,            /**< [OUT] Block number */
  float *line,           /**< [OUT] Line */
  float *sample          /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  double som_x;			/* SOM X */
  double som_y;			/* SOM Y */

  if (block == NULL || line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkLatLonToSomXY(path, lat_dd, lon_dd, &som_x, &som_y);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToBls(path, resolution_meters, som_x, som_y,
			 block, line, sample);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
