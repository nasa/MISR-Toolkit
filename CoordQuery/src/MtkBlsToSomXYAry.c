/*===========================================================================
=                                                                           =
=                             MtkBlsToSomXYAry                              =
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

/** \brief Convert array from Block, Line, Sample, to SOM Coordinates
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of block, line, sample values for path 1 at 1100 meter resolution to SOM Coordinates.
 *
 *  \code
 *  int block[2] = {22, 122};
 *  float line[2] = {101, 202};
 *  float sample[2] = {22, 122};
 *  double som_x[2];
 *  double som_y[2];
 *
 *  status = MtkBlsToSomXYAry(1, 1100, 2, block, line, sample, som_x, som_y);
 *  \endcode
 */

MTKt_status MtkBlsToSomXYAry(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution Meters */
  int nelement,          /**< [IN] Number of elements */
  const int *block,      /**< [IN] Block */
  const float *line,     /**< [IN] Line */
  const float *sample,   /**< [IN] Sample */
  double *som_x,         /**< [OUT] SOM X */
  double *som_y          /**< [OUT] SOM Y */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int i;			/* Loop index */

  if (block == NULL || line == NULL || sample == NULL ||
      som_x == NULL || som_y == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkPathToProjParam(path, resolution_meters, &pp);
  MTK_ERR_COND_JUMP(status);

  status = misr_init(pp.nblock, pp.nline, pp.nsample,
		     pp.reloffset, pp.ulc, pp.lrc);
  if (status != MTK_SUCCESS)
    MTK_ERR_CODE_JUMP(MTK_MISR_PROJ_INIT_FAILED);

  for (i = 0; i < nelement; i++) {
    status = misrinv(block[i], line[i], sample[i], &som_x[i], &som_y[i]);
    if (status != MTK_SUCCESS)
      MTK_ERR_CODE_JUMP(MTK_MISR_INVERSE_PROJ_FAILED);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
