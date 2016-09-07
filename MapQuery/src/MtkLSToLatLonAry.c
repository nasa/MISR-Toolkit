/*===========================================================================
=                                                                           =
=                             MtkLSToLatLonAry                              =
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
#include "MisrError.h"

/** \brief Convert array of line, sample to array of decimal degrees
 *         latitude and longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of line and sample values to latitude and longitude in decimal degrees.
 *
 *  \code
 *  float line[2] = {1.0, 2.0};
 *  float sample[2] = {1.0, 2.0};
 *  double lat_dd[2];
 *  double lon_dd[2];
 *
 *  status = MtkLSToLatLonAry(mapinfo, 2, line, sample, lat_dd, lon_dd);
 *  \endcode
 */

MTKt_status MtkLSToLatLonAry(
  MTKt_MapInfo mapinfo, /**< [IN] Map info */
  int nelement,         /**< [IN] Number of elements */
  const float *line,    /**< [IN] Line */
  const float *sample,  /**< [IN] Sample */
  double *lat_dd,       /**< [OUT] Latitude */
  double *lon_dd        /**< [OUT] Longitude */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  int i;                        /* Loop index */

  if (line == NULL || sample == NULL || lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status_code = MTK_SUCCESS;

  for (i = 0; i < nelement; i++) {
    status = MtkLSToLatLon(mapinfo, line[i], sample[i], &lat_dd[i], &lon_dd[i]);
    if (status) status_code = status;
  }

  return status_code;

ERROR_HANDLE:
  return status_code;
}
