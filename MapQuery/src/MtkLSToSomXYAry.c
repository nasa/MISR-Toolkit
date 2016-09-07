/*===========================================================================
=                                                                           =
=                             MtkLSToSomXYAry                               =
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

/** \brief Convert array of line, sample to array of SOM X, SOM Y
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of line, sample values to SOM Coordinates.
 *
 *  \code
 *  float line[2] = {101, 202};
 *  float sample[2] = {22, 122};
 *  double som_x[2];
 *  double som_y[2];
 *
 *  status = MtkLSToSomXYAry(mapinfo, 2, line, sample, som_x, som_y);
 *  \endcode
 */

MTKt_status MtkLSToSomXYAry(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  int nelement,         /**< [IN] Number of elemnts */
  const float *line,    /**< [IN] Line */
  const float *sample,  /**< [IN] Sample */
  double *som_x,        /**< [OUT] SOM X */
  double *som_y         /**< [OUT] SOM Y */ ) {

  MTKt_status status;		/* Return status */
  MTKt_status status_code;	/* Error return status code */
  int i;                        /* Loop index */

  if (line == NULL || sample == NULL || som_x == NULL || som_y == NULL)
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
    status = MtkLSToSomXY(mapinfo, line[i], sample[i], &som_x[i], &som_y[i]);
    if (status) status_code = status;
  }

  return status_code;

ERROR_HANDLE:
  return status_code;
}
