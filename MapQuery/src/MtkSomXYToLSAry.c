/*===========================================================================
=                                                                           =
=                             MtkSomXYToLSAry                               =
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

/** \brief Convert array of SOM X, SOM Y to array of line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of SOM X and SOM Y values to line and sample.
 *
 *  \code
 *  double som_x[2] = {10529200.016621, 10811900.026208};
 *  double som_y[2] = {622600.018066, 623700.037609};
 *  float line[2];
 *  float sample[2];
 *
 *  status = MtkSomXYToBlsAry(mapinfo, 2, som_x, som_y, line, sample);
 *  \endcode
 */

MTKt_status MtkSomXYToLSAry(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  int nelement,         /**< [IN] Number of elements */
  const double *som_x,  /**< [IN] SOM X */
  const double *som_y,  /**< [IN] SOM Y */
  float *line,          /**< [OUT] Line */
  float *sample         /**< [OUT] Sample */ )
{
  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Error return status code */
  int i;                        /* Loop index */

  if (som_x == NULL || som_y == NULL || line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  /* This routine will processing all elements in the input arrays */
  /* no matter if an error occurred in any of the elements. */
  /* An unsuccessful error code will be return only if one of the */
  /* elements encountered an error.  If none do then the status is */
  /* success. */

  status_code = MTK_SUCCESS;

  for (i = 0; i < nelement; i++) {
    status = MtkSomXYToLS(mapinfo, som_x[i], som_y[i], &line[i], &sample[i]);
    if (status) status_code = status;
  }

  return status_code;

ERROR_HANDLE:
  return status_code;
}
