/*===========================================================================
=                                                                           =
=                            MtkDegMinSecToDms                              =
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
#include <stdlib.h>

/** \brief Convert unpacked Degrees, minutes, seconds to packed
 *
 *  \return MTK_SUCCESS if successful.
 *
 *  \par Example:
 *  In this example, we convert 130 degress, 4 minutes, and 58.23 seconds to packed degrees, minutes, seconds.
 *
 *  \code
 *  status = MtkDegMinSecToDms(130, 4, 58.23, &dms);
 *  \endcode

 */

MTKt_status MtkDegMinSecToDms(
  int deg,    /**< [IN] Degrees */
  int min,    /**< [IN] Minutes */
  double sec, /**< [IN] Seconds */
  double *dms /**< [OUT] Packed degrees, minutes, seconds */ )
{

  MTKt_status status_code;      /* Return status of this function */
  int sgn;			/* Sign of input */

  if (dms == NULL)
    MTK_ERR_CODE_JUMP(MTK_NULLPTR);

  /* determine sign */

  if (deg < 0) {
    sgn = -1;
  } else {
    sgn = 1;
  }
  deg = abs(deg);

  /* Check degrees, minutes, seconds bounds */

  if (deg > 360) return MTK_OUTBOUNDS;
  if (min > 60) return MTK_OUTBOUNDS;
  if (sec > 60.0) return MTK_OUTBOUNDS;

  /* Deg, min, sec to dms */

  *dms = sgn * (deg * 1000000.0 + min * 1000.0 + sec);

  return MTK_SUCCESS;

ERROR_HANDLE:
  return status_code;
}
