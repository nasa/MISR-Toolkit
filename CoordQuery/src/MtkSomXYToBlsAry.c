/*===========================================================================
=                                                                           =
=                             MtkSomXYToBlsAry                              =
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

/** \brief Convert array of SOM X, SOM Y to array of block, line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of SOM X and SOM Y values for path 1 at 1100 meter resolution to block, line, sample.
 *
 *  \code
 *  double som_x[2] = {10529200.016621, 10811900.026208};
 *  double som_y[2] = {622600.018066, 623700.037609};
 *  int block[2];
 *  float line[2];
 *  float sample[2];
 *
 *  status = MtkSomXYToBlsAry(1, 1100, 2, som_x, som_y, block, line, sample);
 *  \endcode
 */

MTKt_status MtkSomXYToBlsAry(
  int path,              /**< [IN] Path */
  int resolution_meters, /**< [IN] Resolution */
  int nelement,          /**< [IN] Number of elements */
  const double *som_x,   /**< [IN] SOM X */
  const double *som_y,   /**< [IN] SOM Y */
  int *block,            /**< [OUT] Block */
  float *line,           /**< [OUT] Line */
  float *sample          /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int i;			/* Loop index */

  if (som_x == NULL || som_y == NULL || block == NULL ||
      line == NULL || sample == NULL)
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
    status = misrfor(som_x[i], som_y[i], &block[i], &line[i], &sample[i]);
    if (status != MTK_SUCCESS)
      MTK_ERR_CODE_JUMP(MTK_MISR_FORWARD_PROJ_FAILED);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
