/*===========================================================================
=                                                                           =
=                             MtkLatLonToLSAry                              =
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
#include "MisrCoordQuery.h"
#include "MisrUnitConv.h"
#include "MisrError.h"
#include "proj.h"
#include "gctp_prototypes.h"
#include "math.h"

/** \brief Convert array of decimal degrees latitude and longitude to
 *         array of  line, sample
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert an array of latitude and longitude to line, sample.
 *
 *  \code
 *  double lat_dd[2] = {44.322089, 45.0};
 *  dobule lon_dd[2] = {-112.00821, -113.0};
 *  float line[2];
 *  float sample[2];
 *
 *  status = MtkLatLonToLSAry(mapinfo, 2, lat_dd, lon_dd, line, sample);
 *  \endcode
 */

MTKt_status MtkLatLonToLSAry(
  MTKt_MapInfo mapinfo, /**< [IN] Map Info */
  int nelement,         /**< [IN] Number of elements */
  const double *lat_dd, /**< [IN] Latitude */
  const double *lon_dd, /**< [IN] Longitude */
  float *line,          /**< [OUT] Line */
  float *sample         /**< [OUT] Sample */ )
{
  MTKt_status status_code;      /* Return status of this function */
  MTKt_status status;           /* Return status */
  int i;                        /* Loop index */
  MTKt_MisrProjParam pp;	/* Projection parameters */
  int iflg;			/* GCTP status flag */
  int (*for_trans[MAXPROJ+1])(double, double, double *, double *); /* Array of function ptrs (Not used) */
  double deg2rad = acos(-1) / 180.0;

  if (lat_dd == NULL || lon_dd == NULL || line == NULL || sample == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  if (nelement < 0)
    MTK_ERR_CODE_JUMP(MTK_BAD_ARGUMENT);

  status_code = MTK_SUCCESS;

  status = MtkPathToProjParam(mapinfo.path, 0, &pp);
  MTK_ERR_COND_JUMP(status);

  for_init(pp.projcode, pp.zonecode, pp.projparam, pp.spherecode,
	   NULL, NULL, &iflg, for_trans);
  if (iflg)
    MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_INIT_FAILED);

  for (i = 0; i < nelement; i++) {
    double som_x;                 /* SOM X */
    double som_y;                 /* SOM Y */
    float tline;
    float tsample; 

    iflg = somfor(lon_dd[i]*deg2rad, lat_dd[i]*deg2rad, &som_x, &som_y);
    if (iflg)
      MTK_ERR_CODE_JUMP(MTK_GCTP_FORWARD_PROJ_FAILED);

    tline = (float)((som_x - mapinfo.som.ulc.x) / mapinfo.resolution);
    tsample = (float)((som_y - mapinfo.som.ulc.y) / mapinfo.resolution);

    /* Check line/sample bounds */
    if (tline < -0.5 || tline > mapinfo.nline - 0.5) {
      tline = -1.0;
      tsample = -1.0;
      status_code = MTK_OUTBOUNDS;
    }

    if (tsample < -0.5 || tsample > mapinfo.nsample - 0.5) {
      tline = -1.0;
      tsample = -1.0;
      status_code = MTK_OUTBOUNDS;
    }

    line[i] = tline;
    sample[i] = tsample;
  }

  return status_code;

ERROR_HANDLE:
  return status_code;
}
