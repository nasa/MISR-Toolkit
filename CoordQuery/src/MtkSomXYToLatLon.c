/*===========================================================================
=                                                                           =
=                             MtkSomXYToLatLon                              =
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

/** \brief Convert SOM X, SOM Y to decimal degrees latitude and longitude
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert  SOM X 10529200.016621 and SOM Y 622600.018066 for path 1 to latitude and longitude in decimal degrees.
 *
 *  \code
 *  status = MtkSomXYToLatLon(1, 10529200.016621, 622600.018066, &lat_dd, &lon_dd);
 *  \endcode
 */

MTKt_status MtkSomXYToLatLon(
  int path,       /**< [IN] Path */
  double som_x,   /**< [IN] SOM X */
  double som_y,   /**< [IN] SOM Y */
  double *lat_dd, /**< [OUT] Latitude */
  double *lon_dd  /**< [OUT] Longitude */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int iflg;			/* GCTP status flag */
  double lat_r;			/* Latitude in radians */
  double lon_r;			/* Longitude in radians */
  int (*inv_trans[MAXPROJ+1])(double, double, double*, double*); /* Array of function ptrs (Not used) */

  if (lat_dd == NULL || lon_dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkPathToProjParam(path, 0, &pp);
  MTK_ERR_COND_JUMP(status);

  inv_init(pp.projcode, pp.zonecode, pp.projparam, pp.spherecode,
	   NULL, NULL, &iflg, inv_trans);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_INVERSE_INIT_FAILED);

  iflg = sominv(som_x, som_y, &lon_r, &lat_r);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_INVERSE_PROJ_FAILED);

  status = MtkRadToDd(lat_r, lat_dd);
  MTK_ERR_COND_JUMP(status);

  status = MtkRadToDd(lon_r, lon_dd);
  MTK_ERR_COND_JUMP(status);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
