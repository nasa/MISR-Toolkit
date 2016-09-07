/*===========================================================================
=                                                                           =
=                              MtkSomXYToLS                                 =
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

/** \brief Convert SOM X, SOM Y to line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert SOM X 10529200.016621 and SOM Y 622600.018066 to line, sample.
 *
 *  \code
 *  status = MtkSomXYToBls(mapinfo, 10529200.016621, 622600.018066, &line, &sample);
 *  \endcode
 */

MTKt_status MtkSomXYToLS(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  double som_x,         /**< [IN] SOM X */
  double som_y,         /**< [IN] SOM Y */
  float *line,          /**< [OUT] Line */
  float *sample         /**< [OUT] Sample */ )
{
  MTKt_status status_code;	/* Return status of this function */

  if (line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* Determine line/sample from ulc som x/y and resolution */
  *line = (float)((som_x - mapinfo.som.ulc.x) / mapinfo.resolution);
  *sample = (float)((som_y - mapinfo.som.ulc.y) / mapinfo.resolution);

  /* Check line/sample bounds */
  if (*line < -0.5 || *line > mapinfo.nline - 0.5)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);
  if (*sample < -0.5 || *sample > mapinfo.nsample - 0.5)
    MTK_ERR_CODE_JUMP(MTK_OUTBOUNDS);

  return MTK_SUCCESS;
ERROR_HANDLE:
  if (status_code != MTK_NULLPTR)
  {
    *line = -1.0;
    *sample = -1.0;
  }
  return status_code;
}
