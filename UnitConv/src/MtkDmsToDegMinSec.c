/*===========================================================================
=                                                                           =
=                            MtkDmsToDegMinSec                              =
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

/** \brief Convert packed degrees, minutes, seconds to unpacked
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130004058.23 dms to unpacked degrees, minutes, seconds.
 *
 *  \code
 *  status = MtkDmsToDegMinSec(130004058.23, &deg, &min, &sec);
 *  \endcode
 */

MTKt_status MtkDmsToDegMinSec(
  double dms, /**< [IN] Packed degrees, minutes, seconds */
  int *deg,   /**< [OUT] Degrees */
  int *min,   /**< [OUT] Minutes */
  double *sec /**< [OUT] Seconds */ )
{
  MTKt_status status_code;      /* Return status of this function */
  int sgn;			/* Sign of input */
  double ang;			/* Absolute value dms angle */

  if (deg == NULL || min == NULL || sec == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* determine sign */

  if (dms < 0.0) {
    sgn = -1;
  } else {
    sgn = 1;
  }
  ang = fabs(dms);

  /* find degrees */

  *deg = (int)(ang / 1000000.0);
  if (*deg > 360) return MTK_OUTBOUNDS;

  /* find minutes */

  *min = (int)((ang - *deg * 1000000.0) / 1000.0);
  if (*min > 60) return MTK_OUTBOUNDS;

  /* find seconds */

  *sec = ang - *deg * 1000000.0 - *min * 1000.0;
  if (*sec > 60.0) return MTK_OUTBOUNDS;

  /* Replace sign */

  *deg *= sgn;

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
