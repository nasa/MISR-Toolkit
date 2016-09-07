/*===========================================================================
=                                                                           =
=                                MtkDdToDms                                 =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2005, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrUnitConv.h"
#include "MisrError.h"
#include <math.h>

/** \brief Convert decimal degrees to packed degrees, minutes, seconds
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130.08284167 degrees to packed degrees, minutes, seconds.
 *
 *  \code
 *  status = MtkDdToDms(130.08284167, &dms);
 *  \endcode
 */

MTKt_status MtkDdToDms(
  double dd,  /**< [IN] Decimal degrees */
  double *dms /**< [OUT] Packed degrees, minutes, seconds */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int sgn;			/* Sign of input */
  int deg;			/* Degrees */
  int min;			/* Minutes */
  double sec;			/* Seconds */
  double ang;			/* Absolute value dms angle */

  if (dms == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* determine sign */

  if (dd < 0.0) {
    sgn = -1;
  } else {
    sgn = 1;
  }
  ang = fabs(dd);

  /* find degrees */

  deg = (int)ang;
  if (deg > 360) return MTK_OUTBOUNDS;

  /* find minutes */

  ang = (ang - deg) * 60.0;
  min = (int)ang;
  if (min > 60) return MTK_OUTBOUNDS;

  /* find seconds */

  sec = (ang - min) * 60.0;
   
  if (sec > 60.0) return MTK_OUTBOUNDS;

  /* pack sign, degrees, minutes, seconds into dms */

  *dms = sgn * (deg * 1000000 + min * 1000 + sec);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
