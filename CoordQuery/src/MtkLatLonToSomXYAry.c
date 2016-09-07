/*===========================================================================
=                                                                           =
=                           MtkLatLonToSomXYAry                             =
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
#include "MisrUnitConv.h"
#include "MisrError.h"
#include "proj.h"
#include "gctp_prototypes.h"
#include <stddef.h>

/** \brief Convert array of decimal degrees latitude and longitude to array
 *  of SOM X, SOM Y
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of latitude and longitude values for path 1 at 1100 meter resolution to block, line, sample.
 *
 *  \code
 *  double lat_dd[2] = {82.740690, 81.057280};
 *  double lon_dd[2] = {-3.310459, -16.810591};
 *  double som_x[2];
 *  double som_y[2];
 *
 *  status = MtkLatLonToSomXYAry(1, 1100, 2, lat_dd, lon_dd, som_x, som_y);
 *  \endcode
 */

MTKt_status MtkLatLonToSomXYAry(
  int path,             /**< [IN] Path */
  int nelement,         /**< [IN] Number of elements */
  const double *lat_dd, /**< [IN] Latitude */
  const double *lon_dd, /**< [IN] Longitude */ 
  double *som_x,        /**< [OUT] SOM X */
  double *som_y         /**< [OUT] SOM Y */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int iflg;			/* GCTP status flag */
  double lat_r;			/* Latitude in radians */
  double lon_r;			/* Longitude in radians */
  int i;			/* Loop index */
  int (*for_trans[MAXPROJ+1])(double, double, double *, double *); /* Array of function ptrs (Not used) */

  if (lat_dd == NULL || lon_dd == NULL || som_x == NULL || som_y == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkPathToProjParam(path, 0, &pp);
  MTK_ERR_COND_JUMP(status);

  for_init(pp.projcode, pp.zonecode, pp.projparam, pp.spherecode,
	   NULL, NULL, &iflg, for_trans);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_INIT_FAILED);

  for (i = 0; i < nelement; i++) {
    status = MtkDdToRad(lat_dd[i], &lat_r);
    MTK_ERR_COND_JUMP(status);

    status = MtkDdToRad(lon_dd[i], &lon_r);
    MTK_ERR_COND_JUMP(status);

    iflg = somfor(lon_r, lat_r, &som_x[i], &som_y[i]);
    if (iflg)
      MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_PROJ_FAILED);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
