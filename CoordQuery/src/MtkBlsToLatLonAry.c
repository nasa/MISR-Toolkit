/*===========================================================================
=                                                                           =
=                            MtkBlsToLatLonAry                              =
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

/** \brief Convert array from Block, Line, Sample, to Latitude and Longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of block, line, sample values for path 1 at 1100 meter resolution to latitude and longitude in decimal degrees.
 *
 *  \code
 *  int block[2] = {22, 24};
 *  float line[2] = {101, 102};
 *  float sample[2] = {22, 23};
 *  double lat_dd[2];
 *  double lon_dd[2];
 *
 *  status = MtkBlsToLatLonAry(1, 1100, 2, block, line, sample, lat_dd, lon_dd);
 *  \endcode
 */

MTKt_status MtkBlsToLatLonAry(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  int nelement,          /**< [IN] Number of elements */
  const int *block,      /**< [IN] Block */
  const float *line,     /**< [IN] Line */
  const float *sample,   /**< [IN] Sample */
  double *lat_dd,        /**< [OUT] Latitude Decimal Degrees */
  double *lon_dd         /**< [OUT] Longitude Decimal Degrees */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  double som_x;			/* SOM X */
  double som_y;			/* SOM Y */
  int i;			/* Loop index */

  if (block == NULL || line == NULL || sample == NULL ||
      lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  for (i = 0; i < nelement; i++) {
    status = MtkBlsToSomXY(path, resolution_meters,
			   block[i], line[i], sample[i], &som_x, &som_y);
    MTK_ERR_COND_JUMP(status);

    status = MtkSomXYToLatLon(path, som_x, som_y, &lat_dd[i], &lon_dd[i]);
    MTK_ERR_COND_JUMP(status);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
