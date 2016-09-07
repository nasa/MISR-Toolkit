/*===========================================================================
=                                                                           =
=                           MtkSomXYToLatLonAry                             =
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

/** \brief Convert array of SOM X, SOM Y to array of latitude, longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of SOM X and SOM Y values for path 1 to latitude and longitude in decimal degrees.
 *
 *  \code
 *  double som_x[2] = {10529200.016621, 10811900.026208};
 *  double som_y[2] = {622600.018066, 623700.037609};
 *  double lat_dd[2];
 *  double lon_dd[2];
 *
 *  status = MtkSomXYToLatLonAry(1, 2, som_x, som_y, lat_dd, lon_dd);
 *  \endcode
 */

MTKt_status MtkSomXYToLatLonAry(
  int path,            /**< [IN] Path */
  int nelement,        /**< [IN] Number of elements */
  const double *som_x, /**< [IN] SOM X */
  const double *som_y, /**< [IN] SOM Y */
  double *lat_dd,      /**< [OUT] Latitude */
  double *lon_dd       /**< [OUT] Longitude */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int iflg;			/* GCTP status flag */
  double lat_r;			/* Latitude in radians */
  double lon_r;			/* Longitude in radians */
  int i;			/* Loop index */
  int (*inv_trans[MAXPROJ+1])(double, double, double*, double*); /* Array of function ptrs (Not used) */

  if (som_x == NULL || som_y == NULL || lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status = MtkPathToProjParam(path, 0, &pp);
  MTK_ERR_COND_JUMP(status);

  inv_init(pp.projcode, pp.zonecode, pp.projparam, pp.spherecode,
	   NULL, NULL, &iflg, inv_trans);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_INVERSE_INIT_FAILED);

  for (i = 0; i < nelement; i++) {
    iflg = sominv(som_x[i], som_y[i], &lon_r, &lat_r);
    if (iflg) 
      MTK_ERR_CODE_JUMP(MTK_GCTP_INVERSE_PROJ_FAILED);

    status = MtkRadToDd(lat_r, &lat_dd[i]);
    MTK_ERR_COND_JUMP(status);

    status = MtkRadToDd(lon_r, &lon_dd[i]);
    MTK_ERR_COND_JUMP(status);
  }

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
