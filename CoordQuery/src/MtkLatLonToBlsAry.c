/*===========================================================================
=                                                                           =
=                            MtkLatLonToBlsAry                              =
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

/** \brief Convert array of decimal degrees latitude and longitude to array
 *         of block, line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of latitude and longitude values for path 1 at 1100 meter resolution to block, line, sample.
 *
 *  \code
 *  double lat_dd[2] = {82.740690, 81.057280};
 *  double lon_dd[2] = {-3.310459, -16.810591};
 *  int block[2];
 *  float line[2];
 *  float sample[2];
 *
 *  status = MtkLatLonToBlsAry(1, 1100, 2, lat_dd, lon_dd, block, line, sample);
 *  \endcode
 */

MTKt_status MtkLatLonToBlsAry(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  int nelement,          /**< [IN] Number of elements */
  const double *lat_dd,  /**< [IN] Latitude */
  const double *lon_dd,  /**< [IN] Longitude */
  int *block,            /**< [OUT] Block number */
  float *line,           /**< [OUT] Line */
  float *sample          /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  double som_x;			/* SOM X */
  double som_y;			/* SOM Y */
  int i;			/* Loop index */

  if (lat_dd == NULL || lon_dd == NULL || block == NULL ||
      line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  for (i = 0; i < nelement; i++) {
    status = MtkLatLonToSomXY(path, lat_dd[i], lon_dd[i], &som_x, &som_y);
    MTK_ERR_COND_JUMP(status);

    status = MtkSomXYToBls(path, resolution_meters, som_x, som_y,
			   &block[i], &line[i], &sample[i]);
    MTK_ERR_COND_JUMP(status);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
