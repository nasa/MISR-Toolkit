/*===========================================================================
=                                                                           =
=                              MtkBlsToLatLon                               =
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
#include "misrproj.h"

/** \brief Convert from Block, Line, Sample, to Latitude and Longitude in decimal degrees
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert block 22, line 101, sample 22 for path 1 at 1100 meter resolution to latitude and longitude in decimal degrees.
 *
 *  \code
 *  status = MtkBlsToLatLon(1, 1100, 22, 101, 22, &lat_dd, &lon_dd);
 *  \endcode
 */

MTKt_status MtkBlsToLatLon(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  int block,             /**< [IN] Block Number */
  float line,            /**< [IN] Line */
  float sample,          /**< [IN] Sample */
  double *lat_dd,  /**< [OUT] Latitude Decimal Degrees */
  double *lon_dd   /**< [OUT] Longitude Decimal Degrees */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  double som_x;			/* SOM X */
  double som_y;			/* SOM Y */

  if (lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkBlsToSomXY(path, resolution_meters, block, line, sample,
			 &som_x, &som_y);
  MTK_ERR_COND_JUMP(status);

  status = MtkSomXYToLatLon(path, som_x, som_y, lat_dd, lon_dd);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
