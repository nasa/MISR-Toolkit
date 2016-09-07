/*===========================================================================
=                                                                           =
=                             MtkLatLonToSomXY                              =
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



/** \brief Convert decimal degrees latitude and longitude to SOM X, SOM Y
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert latitude 82.740690 and longitude -3.310459 for path 1 to SOM Cordinates.
 *
 *  \code
 *  status = MtkLatLonToSomXY(1, 82.740690, -3.310459, &som_x, &som_y);
 *  \endcode
 */

MTKt_status MtkLatLonToSomXY(
  int path,      /**< [IN] Path */
  double lat_dd, /**< [IN] Latitude */
  double lon_dd, /**< [IN] Longitude */
  double *som_x, /**< [OUT] SOM X */
  double *som_y  /**< [OUT] SOM Y */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;		/* Return status */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int iflg;			/* GCTP status flag */
  double lat_r;			/* Latitude in radians */
  double lon_r;			/* Longitude in radians */
  int (*for_trans[MAXPROJ+1])(double, double, double *, double *); /* Array of function ptrs (Not used) */

  if (som_x == NULL || som_y == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  status = MtkPathToProjParam(path, 0, &pp);
  MTK_ERR_COND_JUMP(status);

  for_init(pp.projcode, pp.zonecode, pp.projparam, pp.spherecode,
	   NULL, NULL, &iflg, for_trans);

  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_INIT_FAILED);

  status = MtkDdToRad(lat_dd, &lat_r);
  MTK_ERR_COND_JUMP(status);

  status = MtkDdToRad(lon_dd, &lon_r);
  MTK_ERR_COND_JUMP(status);

  iflg = somfor(lon_r, lat_r, som_x, som_y);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_PROJ_FAILED);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
