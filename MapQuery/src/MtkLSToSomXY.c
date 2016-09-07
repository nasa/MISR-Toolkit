/*===========================================================================
=                                                                           =
=                              MtkLSToSomXY                                 =
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

/** \brief Convert line, sample to SOM X, SOM Y
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert line 1.0 and sample 1.0 to SOM Cordinates.
 *
 *  \code
 *  status = MtkLSToSomXY(mapinfo, 1.0, 1.0, &som_x, &som_y);
 *  \endcode

 */

MTKt_status MtkLSToSomXY(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  float line,           /**< [IN] Line */
  float sample,         /**< [IN] Sample */
  double *som_x,        /**< [OUT] SOM X */
  double *som_y         /**< [OUT] SOM Y */ )
{
  MTKt_status status_code;	/* Return status of this function */

  if (som_x == NULL || som_y == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Check argument bounds */
  if (line < -0.5 || line > mapinfo.nline - 0.5)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (sample < -0.5 || sample > mapinfo.nsample - 0.5)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  /* Determine som x/y from ulc and line/sample and resolution */
  *som_x = mapinfo.som.ulc.x + line * mapinfo.resolution;
  *som_y = mapinfo.som.ulc.y + sample * mapinfo.resolution;

  return MTK_SUCCESS;

ERROR_HANDLE:
  if (status_code != MTK_NULLPTR)
  {
    *som_x = -1e-9;
    *som_y = -1e-9;
  }
  return status_code;
}
