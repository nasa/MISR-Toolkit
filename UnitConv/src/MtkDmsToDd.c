/*===========================================================================
=                                                                           =
=                               MtkDmsToDd                                  =
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

/** \brief Convert packed degrees, minutes, seconds to decimal degrees
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130004058.23 dms to decimal degrees.
 *
 *  \code
 *  status = MtkDmsToDd(130004058.23, &dd);
 *  \endcode
 */

MTKt_status MtkDmsToDd(
  double dms, /**< [IN] Packed degrees, minutes, seconds */
  double *dd  /**< [OUT] Decimal degrees */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int sgn;			/* Sign of input */
  int deg;			/* Degrees */
  int min;			/* Minutes */
  double sec;			/* Seconds */
  double ang;			/* Absolute value dms angle */

  if (dd == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* determine sign */

  if (dms < 0.0) {
    sgn = -1;
  } else {
    sgn = 1;
  }
  ang = fabs(dms);

  /* find degrees */

  deg = (int)(ang / 1000000.0);
  if (deg > 360) return MTK_OUTBOUNDS;

  /* find minutes */

  min = (int)((ang - deg * 1000000.0) / 1000.0);
  if (min > 60) return MTK_OUTBOUNDS;

  /* find seconds */

  sec = ang - deg * 1000000.0 - min * 1000.0;
  if (sec > 60.0) return MTK_OUTBOUNDS;

  /* pack sign, degrees, minutes, seconds into dd */

  *dd = sgn * (deg * 3600.0 + min * 60.0 + sec) / 3600.0;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
