/*===========================================================================
=                                                                           =
=                               MtkSomXYToBls                               =
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

/** \brief Convert SOM X, SOM Y to block, line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert SOM X 10529200.016621 and SOM Y 622600.018066 for path 1 at 1100 meter resolution to block, line, sample.
 *
 *  \code
 *  status = MtkSomXYToBls(1, 1100, 10529200.016621, 622600.018066, &block, &line, &sample);
 *  \endcode
 */

MTKt_status MtkSomXYToBls(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  double som_x,          /**< [IN] SOM X */
  double som_y,          /**< [IN] SOM Y */
  int *block,            /**< [OUT] Block */
  float *line,           /**< [OUT] Line */
  float *sample          /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */

  if (block == NULL || line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkPathToProjParam(path, resolution_meters, &pp);
  MTK_ERR_COND_JUMP(status);

  status = misr_init(pp.nblock, pp.nline, pp.nsample,
		     pp.reloffset, pp.ulc, pp.lrc);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(MTK_MISR_PROJ_INIT_FAILED);

  status = misrfor(som_x, som_y, block, line, sample);
  if (status != MTK_SUCCESS) 
    MTK_ERR_CODE_JUMP(MTK_MISR_FORWARD_PROJ_FAILED);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
