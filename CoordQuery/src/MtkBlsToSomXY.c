/*===========================================================================
=                                                                           =
=                              MtkBlsToSomXY                                =
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

/** \brief Convert from Block, Line, Sample, to SOM Coordinates
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert block 22, line 101, sample 22 for path 1 at 1100 meter resolution to SOM Coordinates.
 *
 *  \code
 *  status = MtkBlsToSomXY(1, 1100, 22, 101, 22, &som_x, &som_y);
 *  \endcode
 */

MTKt_status MtkBlsToSomXY(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution Meters */
  int block,             /**< [IN] Block */
  float line,            /**< [IN] Line */
  float sample,          /**< [IN] Sample */
  double *som_x,         /**< [OUT] SOM X */
  double *som_y          /**< [OUT] SOM Y */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */

  if (som_x == NULL || som_y == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkPathToProjParam(path, resolution_meters, &pp);
  MTK_ERR_COND_JUMP(status);

  status = misr_init(pp.nblock, pp.nline, pp.nsample,
		     pp.reloffset, pp.ulc, pp.lrc);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(MTK_MISR_PROJ_INIT_FAILED);

  status = misrinv(block, line, sample, som_x, som_y);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(MTK_MISR_INVERSE_PROJ_FAILED);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
